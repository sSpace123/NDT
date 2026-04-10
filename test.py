import os
import numpy as np
import torch
import argparse
from config import SAVE_DIR, REGION_NAMES, denormalize_coord
from dataset import get_dataloaders
from model import PINNDamageLocator
from vis import plot_localization_scatter, plot_attention_topology


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='GNN-SHM 测试脚本')
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-show", action="store_true", help="不弹出图像窗口")
    args = parser.parse_args()

    show = not args.no_show

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    model_path = os.path.join(SAVE_DIR, "best_model.pt")
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    model = PINNDamageLocator().to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ck.get('epoch', '?')}, "
          f"val_rmse={ck.get('val_rmse', '?')}")

    # build_splits 会打印 Train/Val/Test 的划分详情并做 assert 验证无交集
    _, _, test_loader = get_dataloaders()
    print(f"\nTest set: {len(test_loader.dataset)} samples "
          f"(× repeat={test_loader.dataset.repeat})")

    all_preds, all_targets, all_logvars, all_attns = [], [], [], []
    all_gt_regions, all_pred_regions = [], []
    errors = []
    correct, total = 0, 0

    for data, coords_norm, cls_labels in test_loader:
        data = data.to(device)
        coords_norm = coords_norm.to(device)
        cls_labels = cls_labels.to(device)

        reg_out, cls_logits, edge_attn = model(data)

        pred_cls = cls_logits.argmax(1)
        correct += (pred_cls == cls_labels).sum().item()
        total += cls_labels.size(0)

        p_mm = denormalize_coord(reg_out[:, :2].cpu().numpy())
        t_mm = denormalize_coord(coords_norm.cpu().numpy())

        all_preds.append(p_mm)
        all_targets.append(t_mm)
        all_logvars.append(reg_out[:, 2].cpu().numpy())
        all_attns.append(edge_attn.cpu().numpy())
        all_gt_regions.extend(cls_labels.cpu().tolist())
        all_pred_regions.extend(pred_cls.cpu().tolist())
        errors.extend(np.linalg.norm(p_mm - t_mm, axis=1))

    if not errors:
        print("No test samples found.")
        return

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_logvars = np.concatenate(all_logvars)
    all_attns = np.concatenate(all_attns)
    errors = np.array(errors)

    acc = correct / max(total, 1) * 100
    avg_std = np.mean(np.exp(0.5 * all_logvars))

    # ========== 打印结果 ==========
    print(f"\n{'='*55}")
    print(f"  Test Results ({total} samples)")
    print(f"{'='*55}")
    print(f"  Classification Accuracy : {acc:.1f}%")
    print(f"  MAE                     : {np.mean(errors):.2f} mm")
    print(f"  RMSE                    : {np.sqrt(np.mean(errors**2)):.2f} mm")
    print(f"  Max Error               : {np.max(errors):.2f} mm")
    print(f"  Avg Confidence (±σ)     : ±{avg_std:.2f} mm")
    print(f"{'='*55}")

    # 逐样本明细
    print(f"\n{'─'*65}")
    print(f"  {'#':>3}  {'GT Region':<8} {'Pred Region':<12} "
          f"{'GT(x,y)':>16} {'Pred(x,y)':>16}  {'Err':>7}")
    print(f"{'─'*65}")
    for i in range(len(errors)):
        gt_r = REGION_NAMES[all_gt_regions[i]]
        pd_r = REGION_NAMES[all_pred_regions[i]]
        ok = "✓" if all_gt_regions[i] == all_pred_regions[i] else "✗"
        gt_xy = f"({all_targets[i,0]:6.1f},{all_targets[i,1]:6.1f})"
        pd_xy = f"({all_preds[i,0]:6.1f},{all_preds[i,1]:6.1f})"
        print(f"  {i:3d}  {gt_r:<8} {pd_r:<8} {ok}   "
              f"{gt_xy:>16} {pd_xy:>16}  {errors[i]:6.1f}mm")
    print(f"{'─'*65}")

    # ========== 绘图 ==========
    os.makedirs(SAVE_DIR, exist_ok=True)

    scatter_path = os.path.join(SAVE_DIR, "test_scatter.png")
    plot_localization_scatter(
        all_targets, all_preds, log_vars=all_logvars,
        accuracy=acc, mean_std=avg_std,
        save_path=scatter_path, show=show)
    print(f"[SAVED] {scatter_path}")

    attn_path = os.path.join(SAVE_DIR, "test_attention.png")
    # 使用所有测试样本的平均 attention
    mean_attn = np.mean(all_attns, axis=0)
    plot_attention_topology(
        mean_attn, model.gnn.edges,
        save_path=attn_path, show=show)
    print(f"[SAVED] {attn_path}")


if __name__ == "__main__":
    main()
