import os
import numpy as np
import torch
import argparse
from config import SAVE_DIR, REGION_DIRS, denormalize_coord
from dataset import get_dataloaders
from model import PINNDamageLocator
from loss import GeometricPINNLoss
from vis import plot_localization_scatter, plot_attention_topology


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='GNN-SHM 测试脚本 (纯回归)')
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

    _, _, test_loader = get_dataloaders()
    print(f"\nTest set: {len(test_loader.dataset)} samples "
          f"(× repeat={test_loader.dataset.repeat})")

    loss_fn = GeometricPINNLoss()
    all_preds, all_targets, all_attns = [], [], []

    for data, coords_norm in test_loader:
        data = data.to(device)
        coords_norm = coords_norm.to(device)

        reg_out, edge_attn = model(data)

        p_mm = denormalize_coord(reg_out.cpu().numpy())
        t_mm = denormalize_coord(coords_norm.cpu().numpy())

        all_preds.append(p_mm)
        all_targets.append(t_mm)
        all_attns.append(edge_attn.cpu().numpy())

    if not all_preds:
        print("No test samples found.")
        return

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_attns = np.concatenate(all_attns)
    errors = np.linalg.norm(all_preds - all_targets, axis=1)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    sr10 = np.mean(errors < 10.0) * 100
    sr20 = np.mean(errors < 20.0) * 100

    # ========== 打印结果 ==========
    print(f"\n{'='*55}")
    print(f"  Test Results ({len(errors)} samples)")
    print(f"{'='*55}")
    print(f"  MAE                : {mae:.2f} mm")
    print(f"  RMSE               : {rmse:.2f} mm")
    print(f"  Max Error          : {np.max(errors):.2f} mm")
    print(f"  Success Rate <10mm : {sr10:.1f}%")
    print(f"  Success Rate <20mm : {sr20:.1f}%")
    print(f"{'='*55}")

    # 逐样本明细
    print(f"\n{'─'*60}")
    print(f"  {'#':>3}  {'GT(x,y)':>18} {'Pred(x,y)':>18}  {'Err':>7}  {'Status'}")
    print(f"{'─'*60}")
    for i in range(len(errors)):
        gt_xy = f"({all_targets[i,0]:6.1f}, {all_targets[i,1]:6.1f})"
        pd_xy = f"({all_preds[i,0]:6.1f}, {all_preds[i,1]:6.1f})"
        ok = "✓" if errors[i] < 20.0 else "✗"
        print(f"  {i:3d}  {gt_xy:>18} {pd_xy:>18}  {errors[i]:6.1f}mm  {ok}")
    print(f"{'─'*60}")

    # ========== 绘图 ==========
    os.makedirs(SAVE_DIR, exist_ok=True)

    scatter_path = os.path.join(SAVE_DIR, "test_scatter.png")
    plot_localization_scatter(
        all_targets, all_preds, mae=mae, sr20=sr20,
        save_path=scatter_path, show=show)
    print(f"[SAVED] {scatter_path}")

    attn_path = os.path.join(SAVE_DIR, "test_attention.png")
    mean_attn = np.mean(all_attns, axis=0)
    plot_attention_topology(
        mean_attn, model.gnn.edges,
        save_path=attn_path, show=show)
    print(f"[SAVED] {attn_path}")


if __name__ == "__main__":
    main()
