import os
import numpy as np
import torch
import argparse
from config import SAVE_DIR, PLOT_DIR, REGION_NAMES, denormalize_coord
from dataset import get_dataloaders
from model import PINNDamageLocator
from vis import plot_localization_scatter

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(SAVE_DIR, "best_model.pt")
    if not os.path.isfile(model_path):
        print(f"[ERROR] Could not find {model_path}.")
        return

    print(f"Loading model from {model_path}")
    model = PINNDamageLocator().to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    _, _, test_loader = get_dataloaders()
    print(f"Test Set Size: {len(test_loader)}")

    all_preds = []
    all_targets = []
    all_log_vars = []
    errors = []
    correct_cls = 0

    for data, coords_norm, cls_labels in test_loader:
        data = data.to(device)
        coords_norm = coords_norm.to(device)
        cls_labels = cls_labels.to(device)

        reg_out, cls_logits, edge_attn = model(data)

        # Classification
        pred_cls = torch.argmax(cls_logits, dim=1)
        correct_cls += (pred_cls == cls_labels).sum().item()

        # Coordinate calculation
        pred_norm = reg_out[:, :2].cpu().numpy()
        target_norm = coords_norm.cpu().numpy()
        
        pred_mm = denormalize_coord(pred_norm)
        target_mm = denormalize_coord(target_norm)
        
        err = np.linalg.norm(pred_mm - target_mm, axis=1)
        errors.extend(err)

        all_preds.append(pred_mm[0])
        all_targets.append(target_mm[0])
        all_log_vars.append(reg_out[:, 2].cpu().numpy()[0])

    if len(errors) == 0:
        print("No test samples found.")
        return

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_log_vars = np.array(all_log_vars)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    acc = (correct_cls / len(test_loader)) * 100.0
    mean_std = np.mean(np.exp(0.5 * all_log_vars))

    print(f"\n{'='*40}")
    print(f"Test Results:")
    print(f"Accuracy: {acc:.1f}%")
    print(f"MAE:      {mae:.2f} mm")
    print(f"RMSE:     {rmse:.2f} mm")
    print(f"Max Err:  {np.max(errors):.2f} mm")
    print(f"Avg Conf: ±{mean_std:.2f} mm")
    print(f"{'='*40}\n")

    plot_localization_scatter(
        all_targets, 
        all_preds, 
        log_vars=all_log_vars,
        accuracy=acc,
        mean_std=mean_std,
        save_path=os.path.join(PLOT_DIR, "test_scatter.png")
    )
    print(f"Saved evaluation scatter plot to {PLOT_DIR}/test_scatter.png")

if __name__ == "__main__":
    main()
