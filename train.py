import os
import argparse
import time
import numpy as np
import torch

from config import (
    SAVE_DIR, EPOCHS, LEARNING_RATE, LAMBDA_REG, LAMBDA_PHYS,
    REGION_DIRS, denormalize_coord,
)
from dataset import get_dataloaders
from model import PINNDamageLocator
from loss import GeometricPINNLoss
from vis import plot_training_curves, plot_localization_scatter, plot_attention_topology


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_mse_mm, n = 0.0, 0.0, 0

    for data, coords_norm in loader:
        data, coords_norm = data.to(device), coords_norm.to(device)

        optimizer.zero_grad()
        reg_out, edge_attn = model(data)
        loss, _ = loss_fn(reg_out, edge_attn, coords_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        with torch.no_grad():
            p_mm = denormalize_coord(reg_out.cpu().numpy())
            t_mm = denormalize_coord(coords_norm.cpu().numpy())
            total_mse_mm += np.mean((p_mm - t_mm) ** 2)

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1), np.sqrt(total_mse_mm / max(n, 1))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets, all_attns = [], [], []

    for data, coords_norm in loader:
        data, coords_norm = data.to(device), coords_norm.to(device)
        reg_out, edge_attn = model(data)
        loss, _ = loss_fn(reg_out, edge_attn, coords_norm)

        all_preds.append(denormalize_coord(reg_out.cpu().numpy()))
        all_targets.append(denormalize_coord(coords_norm.cpu().numpy()))
        all_attns.append(edge_attn.cpu().numpy())
        total_loss += loss.item()
        n += 1

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    attns = np.concatenate(all_attns)
    errors = np.linalg.norm(preds - targets, axis=1)

    return total_loss / max(n, 1), {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(errors),
        'sr20': np.mean(errors < 20.0) * 100,
        'sr10': np.mean(errors < 10.0) * 100,
        'preds': preds, 'targets': targets,
        'errors': errors, 'edge_attns': attns,
    }


def main():
    parser = argparse.ArgumentParser(description='GNN-SHM Geometric PINN Training')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device if args.device != 'auto' else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ld, val_ld, test_ld = get_dataloaders()
    print(f"Batches: train={len(train_ld)} val={len(val_ld)} test={len(test_ld)}")

    model = PINNDamageLocator().to(device)
    loss_fn = GeometricPINNLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    t_losses, v_losses, t_rmses, v_rmses = [], [], [], []
    best_rmse = float('inf')

    print(f"\n{'Ep':>4} {'Loss':>8} {'TrRMSE':>8} {'VaLoss':>8} "
          f"{'VaRMSE':>8} {'MAE':>7} {'<20mm':>6} {'Time':>6}")
    print("-" * 66)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, tr = train_one_epoch(model, train_ld, optimizer, loss_fn, device)
        vl, vr = evaluate(model, val_ld, loss_fn, device)
        scheduler.step()

        t_losses.append(tl); v_losses.append(vl)
        t_rmses.append(tr); v_rmses.append(vr['rmse'])

        mark = ""
        if vr['rmse'] < best_rmse:
            best_rmse = vr['rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse': best_rmse,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            mark = " *"

        print(f"{epoch:4d} {tl:8.4f} {tr:7.1f}mm {vl:8.4f} {vr['rmse']:7.1f}mm "
              f"{vr['mae']:6.1f}mm {vr['sr20']:5.1f}% {time.time()-t0:5.1f}s{mark}")

    print(f"\nTraining done. Best Val RMSE: {best_rmse:.1f}mm")

    plot_training_curves(t_losses, v_losses, t_rmses, v_rmses,
                         save_path=os.path.join(args.save_dir, 'training_curves.png'))

    # ============ 加载 Best Model → 测试 ============
    print("\n--- Test Set Evaluation ---")
    ck = torch.load(os.path.join(args.save_dir, 'best_model.pt'),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f"Loaded best model from epoch {ck['epoch']}")

    _, res = evaluate(model, test_ld, loss_fn, device)

    print(f"\n{'='*50}")
    print(f"  TEST ({len(res['errors'])} samples)")
    print(f"{'='*50}")
    print(f"  MAE          : {res['mae']:.2f} mm")
    print(f"  RMSE         : {res['rmse']:.2f} mm")
    print(f"  Max Error    : {np.max(res['errors']):.2f} mm")
    print(f"  SR <10mm     : {res['sr10']:.1f}%")
    print(f"  SR <20mm     : {res['sr20']:.1f}%")
    print(f"{'='*50}")

    for i, e in enumerate(res['errors']):
        gt = f"({res['targets'][i,0]:6.1f}, {res['targets'][i,1]:6.1f})"
        pd = f"({res['preds'][i,0]:6.1f}, {res['preds'][i,1]:6.1f})"
        print(f"  #{i} GT{gt} Pred{pd} Err={e:.1f}mm")

    plot_localization_scatter(
        res['targets'], res['preds'], mae=res['mae'], sr20=res['sr20'],
        save_path=os.path.join(args.save_dir, 'test_scatter.png'), show=True)
    plot_attention_topology(
        np.mean(res['edge_attns'], axis=0), model.gnn.edges,
        save_path=os.path.join(args.save_dir, 'test_attention.png'), show=True)


if __name__ == '__main__':
    main()
