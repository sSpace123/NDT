import os
import numpy as np
import torch
import argparse
from config import SAVE_DIR, denormalize_coord
from dataset import get_dataloaders
from model import PINNDamageLocator
from vis import plot_localization_scatter


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='GNN-SHM 测试脚本')
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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
    print(f"Loaded model from epoch {ck.get('epoch', '?')}")

    _, _, test_loader = get_dataloaders()

    all_preds, all_targets, all_logvars, errors = [], [], [], []
    correct, total = 0, 0

    for data, coords_norm, cls_labels in test_loader:
        data = data.to(device)
        coords_norm = coords_norm.to(device)
        cls_labels = cls_labels.to(device)

        reg_out, cls_logits, _ = model(data)

        correct += (cls_logits.argmax(1) == cls_labels).sum().item()
        total += cls_labels.size(0)

        p_mm = denormalize_coord(reg_out[:, :2].cpu().numpy())
        t_mm = denormalize_coord(coords_norm.cpu().numpy())

        all_preds.append(p_mm)
        all_targets.append(t_mm)
        all_logvars.append(reg_out[:, 2].cpu().numpy())
        errors.extend(np.linalg.norm(p_mm - t_mm, axis=1))

    if not errors:
        print("No test samples.")
        return

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_logvars = np.concatenate(all_logvars)
    errors = np.array(errors)

    acc = correct / max(total, 1) * 100
    avg_std = np.mean(np.exp(0.5 * all_logvars))

    print(f"\n{'='*40}")
    print(f"Test Results ({total} samples):")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  MAE:      {np.mean(errors):.2f} mm")
    print(f"  RMSE:     {np.sqrt(np.mean(errors**2)):.2f} mm")
    print(f"  Max Err:  {np.max(errors):.2f} mm")
    print(f"  Avg σ:    ±{avg_std:.2f} mm")
    print(f"{'='*40}")

    plot_localization_scatter(
        all_targets, all_preds, log_vars=all_logvars,
        accuracy=acc, mean_std=avg_std,
        save_path=os.path.join(SAVE_DIR, "test_scatter.png"))
    print(f"Saved: {SAVE_DIR}/test_scatter.png")


if __name__ == "__main__":
    main()
