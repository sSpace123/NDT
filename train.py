import os
import argparse
import time
import numpy as np
import torch

from config import (
    SAVE_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, LAMBDA_REG, LAMBDA_PHYS,
    denormalize_coord
)
from dataset import get_dataloaders
from model import PINNDamageLocator
from loss import PINNLoss
from vis import plot_training_curves, plot_localization_scatter, plot_attention_topology

# ========================================
# 训练核心逻辑
# ========================================
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    total_mse_mm = 0.0
    num_batches = 0

    for batch_idx, (data, coords_norm, cls_labels) in enumerate(dataloader):
        data = data.to(device)
        coords_norm = coords_norm.to(device) # [-1, 1]
        cls_labels = cls_labels.to(device)

        optimizer.zero_grad()
        reg_out, cls_logits, edge_attn = model(data)
        
        # Loss calculation in Normalized Domain [-1, 1]
        loss, loss_dict = loss_fn(reg_out, cls_logits, edge_attn, coords_norm, cls_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Metrics converted to Physical Domain (mm) for logging
        with torch.no_grad():
            preds_norm = reg_out[:, :2].detach().cpu().numpy()
            targets_norm = coords_norm.detach().cpu().numpy()
            
            preds_mm = denormalize_coord(preds_norm)
            targets_mm = denormalize_coord(targets_norm)
            batch_mse_mm = np.mean((preds_mm - targets_mm) ** 2)

        total_loss += loss.item()
        total_mse_mm += batch_mse_mm
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Cls: {loss_dict['loss_cls']:.4f} | "
                  f"Reg: {loss_dict['loss_reg']:.4f} | "
                  f"Entropy: {loss_dict['loss_phys']:.4f} | "
                  f"RMSE(mm): {np.sqrt(batch_mse_mm):.2f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = np.sqrt(total_mse_mm / max(num_batches, 1))
    return avg_loss, avg_rmse

@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_mse_mm = 0.0
    num_batches = 0
    correct_cls = 0
    total_samples = 0

    all_preds_mm = []
    all_targets_mm = []
    all_log_vars = []
    all_edge_attns = []

    for data, coords_norm, cls_labels in dataloader:
        data = data.to(device)
        coords_norm = coords_norm.to(device)
        cls_labels = cls_labels.to(device)

        reg_out, cls_logits, edge_attn = model(data)
        loss, loss_dict = loss_fn(reg_out, cls_logits, edge_attn, coords_norm, cls_labels)

        # Classification Accuracy
        preds_cls = torch.argmax(cls_logits, dim=1)
        correct_cls += (preds_cls == cls_labels).sum().item()
        total_samples += cls_labels.size(0)

        # Physical Metric calculation
        preds_norm = reg_out[:, :2].cpu().numpy()
        targets_norm = coords_norm.cpu().numpy()
        
        preds_mm = denormalize_coord(preds_norm)
        targets_mm = denormalize_coord(targets_norm)
        batch_mse_mm = np.mean((preds_mm - targets_mm) ** 2)

        total_loss += loss.item()
        total_mse_mm += batch_mse_mm
        num_batches += 1

        all_preds_mm.append(preds_mm)
        all_targets_mm.append(targets_mm)
        all_log_vars.append(reg_out[:, 2].cpu().numpy())
        all_edge_attns.append(edge_attn.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = np.sqrt(total_mse_mm / max(num_batches, 1))
    accuracy = (correct_cls / max(total_samples, 1)) * 100.0

    results = {
        'preds': np.concatenate(all_preds_mm, axis=0),
        'targets': np.concatenate(all_targets_mm, axis=0),
        'log_vars': np.concatenate(all_log_vars, axis=0),
        'edge_attns': np.concatenate(all_edge_attns, axis=0),
        'accuracy': accuracy
    }

    return avg_loss, avg_rmse, accuracy, results

def main():
    parser = argparse.ArgumentParser(description='GNN-SHM Bipartite Refactor 训练脚本')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--lambda_reg', type=float, default=LAMBDA_REG)
    parser.add_argument('--lambda_phys', type=float, default=LAMBDA_PHYS)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Automatically loads dataloader (it will load dummy if DATA_ROOT has no csv files for quick run)
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"训练集 Batch 数量: {len(train_loader)} | 验证集: {len(val_loader)}")

    model = PINNDamageLocator().to(device)
    loss_fn = PINNLoss(lambda_reg=args.lambda_reg, lambda_phys=args.lambda_phys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []
    best_val_rmse = float('inf')

    print(f"\n开始训练 ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        t_loss, t_rmse = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        train_losses.append(t_loss)
        train_rmses.append(t_rmse)

        v_loss, v_rmse, v_acc, val_results = validate(model, val_loader, loss_fn, device)
        val_losses.append(v_loss)
        val_rmses.append(v_rmse)

        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {t_loss:.4f} RMSE(mm): {t_rmse:.2f} | "
              f"Val Loss: {v_loss:.4f} RMSE(mm): {v_rmse:.2f} Acc: {v_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': v_rmse,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ★ 保存最优模型 (Val RMSE: {v_rmse:.2f}mm)")

    print(f"\n训练完成！最优 Val RMSE: {best_val_rmse:.2f}mm")

    print("生成训练可视化图表...")
    plot_training_curves(
        train_losses, val_losses, train_rmses, val_rmses,
        save_path=os.path.join(args.save_dir, 'training_curves.png')
    )

    if val_results['preds'].shape[0] > 0:
        log_vars = val_results['log_vars']
        avg_std = np.mean(np.exp(0.5 * log_vars))
        
        plot_localization_scatter(
            val_results['targets'],
            val_results['preds'],
            log_vars=log_vars,
            accuracy=val_results['accuracy'],
            mean_std=avg_std,
            save_path=os.path.join(args.save_dir, 'localization_scatter.png')
        )

    if val_results['edge_attns'].shape[0] > 0:
        plot_attention_topology(
            val_results['edge_attns'][0],
            model.spatial_gnn.edges,
            save_path=os.path.join(args.save_dir, 'attention_topology.png')
        )

if __name__ == '__main__':
    main()
