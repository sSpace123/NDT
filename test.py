# -*- coding: utf-8 -*-
"""
test.py  —  评估 (v4.1 — 多模型集成 + TTA)
  - Test-Time Augmentation: 多次随机增强推理, 取均值
  - Multi-Seed Ensemble: 加载多个 seed 模型, 取均值
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import (
    SAVE_DIR, PLOT_DIR, REGION_NAMES, REGION_CENTERS, NUM_REGIONS,
    SENSOR_COORDS, denormalize_coord, COORD_MIN, COORD_MAX,
    WINDOW_LEN, NOISE_STD, NUM_PAIRS,
)
from dataset import build_full_eval_dataset
from model import NDTLocalizer

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BLADE_ZONES = {
    "叶尖 (y>0)": [0, 1, 2],
    "叶中 (y≈0)": [3, 4, 5],
    "叶根 (y<0)": [6, 7, 8],
}

TTA_RUNS = 30  # TTA 推理次数


def load_models(device):
    """加载所有集成模型"""
    info_path = os.path.join(SAVE_DIR, "ensemble_info.json")
    if os.path.isfile(info_path):
        with open(info_path) as f:
            info = json.load(f)
        paths = info["models"]
        print(f"[INFO] 集成: {len(paths)} 个模型")
    else:
        paths = [os.path.join(SAVE_DIR, "best_model.pt")]
        print("[INFO] 单模型模式")

    models = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"  [WARN] 跳过 {p}")
            continue
        m = NDTLocalizer()
        ck = torch.load(p, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        m.to(device).eval()
        seed = ck.get("seed", "?")
        mae = ck.get("val_mae_mm", "?")
        print(f"  模型: Seed={seed}, E{ck.get('epoch','?')}, val_mae={mae}")
        models.append(m)
    return models


def tta_augment(x_orig, di_orig):
    """TTA 增强: 随机时移 + 轻微噪声"""
    x = x_orig.clone()
    B, K, C, W = x.shape
    N_total = W  # 已经裁剪过

    # 小幅随机噪声
    x = x + torch.randn_like(x) * (NOISE_STD * 0.5)

    # 小幅随机缩放
    scale = 0.9 + torch.rand(1).item() * 0.2
    x = x * scale

    # 随机掩蔽 2-3 个传感器对
    n_mask = np.random.randint(1, 4)
    mask_idx = np.random.choice(K, n_mask, replace=False)
    x[:, mask_idx] = 0

    return x, di_orig


@torch.no_grad()
def evaluate_with_tta(models, ds, device, n_tta=TTA_RUNS):
    """
    多模型集成 + TTA:
      对每个样本: 每个模型运行 n_tta 次增强推理
      最终预测 = 所有 (模型 × TTA) 结果的均值
    """
    R = {"gt_mm":[], "pred_mm":[], "gt_r":[], "pred_r":[], "err":[], "std_mm":[]}

    for i in range(len(ds)):
        x, di, ri, cn = ds[i]
        x0 = x.unsqueeze(0).to(device)
        di0 = di.unsqueeze(0).to(device)
        gt = denormalize_coord(cn.numpy())

        all_preds = []
        all_logits = []

        for model in models:
            model.eval()
            # 原始推理 (无增强)
            out = model(x0, di0)
            all_preds.append(out["pred_loc"].cpu().numpy()[0])
            all_logits.append(out["loc_logits"].cpu().numpy()[0])

            # TTA 增强推理
            for _ in range(n_tta):
                x_aug, di_aug = tta_augment(x0, di0)
                out_aug = model(x_aug, di_aug)
                all_preds.append(out_aug["pred_loc"].cpu().numpy()[0])
                all_logits.append(out_aug["loc_logits"].cpu().numpy()[0])

        # 取均值
        mean_pred_norm = np.mean(all_preds, axis=0)
        mean_logits = np.mean(all_logits, axis=0)
        pred_mm = denormalize_coord(mean_pred_norm)
        pred_r = int(np.argmax(mean_logits))

        # 预测标准差 (量化不确定性)
        preds_mm = np.array([denormalize_coord(p) for p in all_preds])
        std_mm = np.mean(np.std(preds_mm, axis=0))

        err = np.linalg.norm(pred_mm - gt)
        R["gt_mm"].append(gt); R["pred_mm"].append(pred_mm)
        R["gt_r"].append(ri.item()); R["pred_r"].append(pred_r)
        R["err"].append(err); R["std_mm"].append(std_mm)

    return {k: np.array(v) for k, v in R.items()}


def print_results(R, n_models, n_tta):
    e = R["err"]
    mae, rmse, mx = np.mean(e), np.sqrt(np.mean(e**2)), np.max(e)
    n_ok = int(np.sum(R["gt_r"] == R["pred_r"]))
    print(f"\n{'='*65}")
    print(f"  集成+TTA: {n_models}模型 × {n_tta+1}次/样本")
    print(f"  全样本: {len(e)}个, {len(set(R['gt_r'].tolist()))}/{NUM_REGIONS}区域")
    print(f"  MAE={mae:.2f}mm  RMSE={rmse:.2f}mm  Max={mx:.2f}mm")
    print(f"  分类: {n_ok}/{len(e)} ({n_ok/len(e):.0%})")
    print(f"{'='*65}")

    print(f"\n{'─'*70}")
    print(f"  {'区域':<5} {'N':>3} {'MAE':>8} {'Max':>8} {'σ':>7} {'分类':>5}")
    print(f"{'─'*70}")
    for r in range(NUM_REGIONS):
        m = R["gt_r"] == r
        if not m.any(): print(f"  {REGION_NAMES[r]:<5}   0"); continue
        er = R["err"][m]; st = R["std_mm"][m]
        c_ok = int(np.sum(R["pred_r"][m] == r))
        print(f"  {REGION_NAMES[r]:<5} {int(m.sum()):>3} {np.mean(er):>8.2f} "
              f"{np.max(er):>8.2f} {np.mean(st):>7.2f} {c_ok}/{int(m.sum()):>3}")
    print(f"{'─'*70}")

    print(f"\n{'═'*55}")
    for zn, ri_list in BLADE_ZONES.items():
        m = np.isin(R["gt_r"], ri_list)
        if m.any(): print(f"  {zn}: MAE={np.mean(R['err'][m]):.2f}mm (n={int(m.sum())})")
    print(f"{'═'*55}")

    print("\n逐样本:")
    for i in range(len(e)):
        g, p = R["gt_mm"][i], R["pred_mm"][i]
        ok = "✓" if R["gt_r"][i] == R["pred_r"][i] else "✗"
        print(f"  [{i:2}] GT=({g[0]:6.1f},{g[1]:6.1f}) Pred=({p[0]:6.1f},{p[1]:6.1f}) "
              f"Err={e[i]:.1f}mm σ={R['std_mm'][i]:.1f} {ok} "
              f"{REGION_NAMES[R['gt_r'][i]]}→{REGION_NAMES[R['pred_r'][i]]}")
    return mae, rmse, mx


def plot_all(R, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    gt, pred, err = R["gt_mm"], R["pred_mm"], R["err"]

    # 1. 2D scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(SENSOR_COORDS[:,0], SENSOR_COORDS[:,1], c="green", marker="^", s=100, zorder=5, label="传感器")
    for r in range(NUM_REGIONS):
        cx, cy = REGION_CENTERS[r]
        ax.add_patch(plt.Rectangle((cx-37.5, cy-37.5), 75, 75, fill=False, ec="gray", lw=0.8, ls="--"))
        ax.text(cx, cy+42, REGION_NAMES[r], ha="center", fontsize=9, color="gray")
    ax.scatter(gt[:,0], gt[:,1], c="red", marker="*", s=250, zorder=10, label="GT", ec="darkred")
    ax.scatter(pred[:,0], pred[:,1], c="dodgerblue", marker="o", s=80, zorder=8, label="Pred", ec="navy", alpha=0.8)
    for i in range(len(err)):
        ax.annotate(f"{err[i]:.1f}", xy=(pred[i,0], pred[i,1]), fontsize=7, xytext=(5,5),
                    textcoords="offset points", color="navy")
    ax.set_xlim(COORD_MIN-20, COORD_MAX+20); ax.set_ylim(COORD_MIN-20, COORD_MAX+20)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title(f"NDT 集成+TTA (MAE={np.mean(err):.1f}mm)", fontsize=14, fontweight="bold")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "prediction_scatter.png"), dpi=150); plt.close()

    # 2. Error vectors
    fig, ax = plt.subplots(figsize=(10, 10))
    for r in range(NUM_REGIONS):
        cx, cy = REGION_CENTERS[r]
        ax.add_patch(plt.Rectangle((cx-37.5, cy-37.5), 75, 75, fill=False, ec="lightgray", lw=0.8, ls="--"))
        ax.text(cx, cy+42, REGION_NAMES[r], ha="center", fontsize=9, color="gray")
    norm = plt.Normalize(0, max(err.max(), 1)); cmap = plt.cm.YlOrRd
    for i in range(len(gt)):
        ax.annotate("", xy=pred[i], xytext=gt[i],
                     arrowprops=dict(arrowstyle="->", color=cmap(norm(err[i])), lw=2))
    ax.scatter(gt[:,0], gt[:,1], c="red", marker="*", s=200, zorder=10, label="GT")
    ax.scatter(pred[:,0], pred[:,1], c="dodgerblue", marker="o", s=60, zorder=8, label="Pred", alpha=0.7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label="误差 (mm)")
    ax.set_xlim(COORD_MIN-20, COORD_MAX+20); ax.set_ylim(COORD_MIN-20, COORD_MAX+20)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title("误差向量图 (集成+TTA)", fontsize=14, fontweight="bold")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "error_vectors.png"), dpi=150); plt.close()

    # 3. Bars
    mae_r = [np.mean(err[R["gt_r"]==r]) if (R["gt_r"]==r).any() else 0 for r in range(NUM_REGIONS)]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, NUM_REGIONS))
    bars = ax.bar(REGION_NAMES, mae_r, color=colors, ec="k", lw=0.5)
    for b, v in zip(bars, mae_r):
        if v > 0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("MAE (mm)"); ax.set_title("各区域MAE (集成+TTA)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "per_region_mae.png"), dpi=150); plt.close()

    # 4. Blade trend
    zn_names, zn_mae = [], []
    for zn, ri_list in BLADE_ZONES.items():
        m = np.isin(R["gt_r"], ri_list)
        if m.any(): zn_names.append(zn.split("(")[0].strip()); zn_mae.append(np.mean(err[m]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(zn_names, zn_mae, color=["#ff9999", "#66b3ff", "#99ff99"], ec="k", lw=0.5)
    for i, v in enumerate(zn_mae): ax.text(i, v+0.3, f"{v:.1f}", ha="center", fontsize=11)
    ax.set_ylabel("MAE (mm)"); ax.set_title("叶尖 vs 叶根 (集成+TTA)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "blade_trend.png"), dpi=150); plt.close()

    # 5. Uncertainty plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_u = ["green" if e < 10 else "orange" if e < 30 else "red" for e in err]
    bars = ax.bar(range(len(err)), err, color=colors_u, ec="k", lw=0.5, alpha=0.7)
    ax.errorbar(range(len(err)), err, yerr=R["std_mm"], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(range(len(err)))
    labels = [f"{REGION_NAMES[int(R['gt_r'][i])]}" for i in range(len(err))]
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("误差 (mm)"); ax.set_xlabel("样本")
    ax.set_title("逐样本误差 + 不确定性 (σ)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "uncertainty.png"), dpi=150); plt.close()

    print(f"[PLOT] 5张图已保存到 {plot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", type=int, default=TTA_RUNS)
    args = parser.parse_args()
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                        else args.device if args.device != "auto" else "cpu")

    models = load_models(dev)
    if not models:
        print("[ERROR] 无可用模型"); sys.exit(1)

    ds, _ = build_full_eval_dataset()
    print(f"[INFO] 样本数: {len(ds)}, TTA={args.tta}次, 集成={len(models)}模型")

    R = evaluate_with_tta(models, ds, dev, n_tta=args.tta)
    mae, rmse, mx = print_results(R, len(models), args.tta)
    plot_all(R, PLOT_DIR)

    with open(os.path.join(PLOT_DIR, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"MAE={mae:.2f}mm RMSE={rmse:.2f}mm Max={mx:.2f}mm\n")
        f.write(f"Ensemble={len(models)}, TTA={args.tta}\n\n")
        for i in range(len(R["err"])):
            f.write(f"[{i}] GT=({R['gt_mm'][i,0]:.1f},{R['gt_mm'][i,1]:.1f}) "
                    f"Pred=({R['pred_mm'][i,0]:.1f},{R['pred_mm'][i,1]:.1f}) "
                    f"Err={R['err'][i]:.2f}mm σ={R['std_mm'][i]:.1f} "
                    f"{REGION_NAMES[R['gt_r'][i]]}→{REGION_NAMES[R['pred_r'][i]]}\n")
    print("[完成]")


if __name__ == "__main__":
    main()
