import os
import argparse
import time
import numpy as np
import torch

from config import SAVE_DIR, EPOCHS, LEARNING_RATE, LAMBDA_REG, LAMBDA_PHYS, denormalize_coord
from dataset import get_dataloaders
from model import PINNDamageLocator
from loss import PINNLoss
from vis import plot_training_curves, plot_localization_scatter, plot_attention_topology


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss, total_mse_mm, n = 0.0, 0.0, 0

    for batch_idx, (data, coords_norm, cls_labels) in enumerate(loader):
        data, coords_norm, cls_labels = (
            data.to(device), coords_norm.to(device), cls_labels.to(device))

        optimizer.zero_grad()
        reg_out, cls_logits, edge_attn = model(data)
        loss, ld = loss_fn(reg_out, cls_logits, edge_attn, coords_norm, cls_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            p_mm = denormalize_coord(reg_out[:, :2].cpu().numpy())
            t_mm = denormalize_coord(coords_norm.cpu().numpy())
            batch_mse = np.mean((p_mm - t_mm) ** 2)

        total_loss += loss.item()
        total_mse_mm += batch_mse
        n += 1

        if batch_idx % 10 == 0:
            print(f"  [E{epoch}] B{batch_idx}/{len(loader)} | "
                  f"L={loss.item():.4f} cls={ld['loss_cls']:.4f} "
                  f"reg={ld['loss_reg']:.4f} ent={ld['loss_entropy']:.4f} "
                  f"RMSE={np.sqrt(batch_mse):.1f}mm")

    return total_loss / max(n, 1), np.sqrt(total_mse_mm / max(n, 1))


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_mse_mm, n = 0.0, 0.0, 0
    correct, total_samples = 0, 0
    all_preds, all_targets, all_logvars, all_attns = [], [], [], []

    for data, coords_norm, cls_labels in loader:
        data, coords_norm, cls_labels = (
            data.to(device), coords_norm.to(device), cls_labels.to(device))

        reg_out, cls_logits, edge_attn = model(data)
        loss, _ = loss_fn(reg_out, cls_logits, edge_attn, coords_norm, cls_labels)

        correct += (cls_logits.argmax(1) == cls_labels).sum().item()
        total_samples += cls_labels.size(0)

        p_mm = denormalize_coord(reg_out[:, :2].cpu().numpy())
        t_mm = denormalize_coord(coords_norm.cpu().numpy())

        total_loss += loss.item()
        total_mse_mm += np.mean((p_mm - t_mm) ** 2)
        n += 1

        all_preds.append(p_mm)
        all_targets.append(t_mm)
        all_logvars.append(reg_out[:, 2].cpu().numpy())
        all_attns.append(edge_attn.cpu().numpy())

    acc = correct / max(total_samples, 1) * 100
    results = {
        'preds': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets),
        'log_vars': np.concatenate(all_logvars),
        'edge_attns': np.concatenate(all_attns),
        'accuracy': acc,
    }
    return total_loss / max(n, 1), np.sqrt(total_mse_mm / max(n, 1)), acc, results


def main():
    parser = argparse.ArgumentParser(description='GNN-SHM 训练脚本')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--lambda_reg', type=float, default=LAMBDA_REG)
    parser.add_argument('--lambda_phys', type=float, default=LAMBDA_PHYS)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device if args.device != 'auto' else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_ld, val_ld, _ = get_dataloaders()
    print(f"Train batches: {len(train_ld)} | Val batches: {len(val_ld)}")

    model = PINNDamageLocator().to(device)
    loss_fn = PINNLoss(lambda_reg=args.lambda_reg, lambda_phys=args.lambda_phys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    t_losses, v_losses, t_rmses, v_rmses = [], [], [], []
    best_rmse = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, tr = train_one_epoch(model, train_ld, optimizer, loss_fn, device, epoch)
        vl, vr, va, vres = validate(model, val_ld, loss_fn, device)
        scheduler.step()

        t_losses.append(tl); v_losses.append(vl)
        t_rmses.append(tr); v_rmses.append(vr)

        print(f"E{epoch:03d}/{args.epochs} "
              f"TrL={tl:.4f} TrRMSE={tr:.1f}mm | "
              f"VL={vl:.4f} VRMSE={vr:.1f}mm Acc={va:.1f}% "
              f"({time.time()-t0:.1f}s)")

        if vr < best_rmse:
            best_rmse = vr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': vr,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ★ Best model saved (RMSE={vr:.1f}mm)")

    print(f"\nDone. Best Val RMSE: {best_rmse:.1f}mm")

    # 可视化
    plot_training_curves(t_losses, v_losses, t_rmses, v_rmses,
                         save_path=os.path.join(args.save_dir, 'training_curves.png'))

    if vres['preds'].shape[0] > 0:
        avg_std = np.mean(np.exp(0.5 * vres['log_vars']))
        plot_localization_scatter(
            vres['targets'], vres['preds'], log_vars=vres['log_vars'],
            accuracy=vres['accuracy'], mean_std=avg_std,
            save_path=os.path.join(args.save_dir, 'localization_scatter.png'))

    if vres['edge_attns'].shape[0] > 0:
        plot_attention_topology(
            vres['edge_attns'][0], model.gnn.edges,
            save_path=os.path.join(args.save_dir, 'attention_topology.png'))


if __name__ == '__main__':
    main()
