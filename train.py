import os
import argparse
import time
import numpy as np
import torch

from config import (
    SAVE_DIR, EPOCHS, LEARNING_RATE, LAMBDA_REG, LAMBDA_PHYS,
    REGION_NAMES, denormalize_coord,
)
from dataset import get_dataloaders
from model import PINNDamageLocator
from loss import PINNLoss
from vis import plot_training_curves, plot_localization_scatter, plot_attention_topology


# ========================================
# 训练核心
# ========================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """静默训练一个 Epoch, 仅返回统计量, 无任何 Batch 级 print"""
    model.train()
    total_loss, total_mse_mm, n = 0.0, 0.0, 0

    for data, coords_norm, cls_labels in loader:
        data, coords_norm, cls_labels = (
            data.to(device), coords_norm.to(device), cls_labels.to(device))

        optimizer.zero_grad()
        reg_out, cls_logits, edge_attn = model(data)
        loss, _ = loss_fn(reg_out, cls_logits, edge_attn, coords_norm, cls_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            p_mm = denormalize_coord(reg_out[:, :2].cpu().numpy())
            t_mm = denormalize_coord(coords_norm.cpu().numpy())
            total_mse_mm += np.mean((p_mm - t_mm) ** 2)

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1), np.sqrt(total_mse_mm / max(n, 1))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """通用评估函数, 供 validation 和 test 共用"""
    model.eval()
    total_loss, total_mse_mm, n = 0.0, 0.0, 0
    correct, total_samples = 0, 0
    all_preds, all_targets, all_logvars, all_attns = [], [], [], []
    all_gt_cls, all_pred_cls = [], []

    for data, coords_norm, cls_labels in loader:
        data, coords_norm, cls_labels = (
            data.to(device), coords_norm.to(device), cls_labels.to(device))

        reg_out, cls_logits, edge_attn = model(data)
        loss, _ = loss_fn(reg_out, cls_logits, edge_attn, coords_norm, cls_labels)

        pred_cls = cls_logits.argmax(1)
        correct += (pred_cls == cls_labels).sum().item()
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
        all_gt_cls.extend(cls_labels.cpu().tolist())
        all_pred_cls.extend(pred_cls.cpu().tolist())

    acc = correct / max(total_samples, 1) * 100
    results = {
        'preds': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets),
        'log_vars': np.concatenate(all_logvars),
        'edge_attns': np.concatenate(all_attns),
        'gt_cls': all_gt_cls,
        'pred_cls': all_pred_cls,
        'accuracy': acc,
    }
    return total_loss / max(n, 1), np.sqrt(total_mse_mm / max(n, 1)), acc, results


# ========================================
# 训测一体入口
# ========================================

def main():
    parser = argparse.ArgumentParser(description='GNN-SHM Train + Auto-Test')
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

    # ============ 数据加载 ============
    train_ld, val_ld, test_ld = get_dataloaders()
    print(f"Batches: train={len(train_ld)} val={len(val_ld)} test={len(test_ld)}")

    model = PINNDamageLocator().to(device)
    loss_fn = PINNLoss(lambda_reg=args.lambda_reg, lambda_phys=args.lambda_phys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    t_losses, v_losses, t_rmses, v_rmses = [], [], [], []
    best_rmse = float('inf')

    # ============ 训练循环 (极简日志) ============
    print(f"\n{'Ep':>4} {'TrLoss':>8} {'TrRMSE':>8} {'VaLoss':>8} {'VaRMSE':>8} {'VaAcc':>7} {'Time':>6}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, tr = train_one_epoch(model, train_ld, optimizer, loss_fn, device)
        vl, vr, va, _ = evaluate(model, val_ld, loss_fn, device)
        scheduler.step()

        t_losses.append(tl); v_losses.append(vl)
        t_rmses.append(tr); v_rmses.append(vr)

        mark = ""
        if vr < best_rmse:
            best_rmse = vr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': vr,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            mark = " *"

        print(f"{epoch:4d} {tl:8.4f} {tr:7.1f}mm {vl:8.4f} {vr:7.1f}mm {va:6.1f}% "
              f"{time.time()-t0:5.1f}s{mark}")

    print(f"\nTraining done. Best Val RMSE: {best_rmse:.1f}mm")

    # ============ 训练曲线 ============
    plot_training_curves(t_losses, v_losses, t_rmses, v_rmses,
                         save_path=os.path.join(args.save_dir, 'training_curves.png'))

    # ============ 自动加载 Best Model -> 测试集推断 -> 出图 ============
    print("\n--- Auto-Evaluating Best Model on Test Set ---")
    ck = torch.load(os.path.join(args.save_dir, 'best_model.pt'),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f"Loaded best model from epoch {ck['epoch']} (val_rmse={ck['val_rmse']:.1f}mm)")

    _, test_rmse, test_acc, test_res = evaluate(model, test_ld, loss_fn, device)
    errors = np.linalg.norm(test_res['preds'] - test_res['targets'], axis=1)
    avg_std = np.mean(np.exp(0.5 * test_res['log_vars']))

    # 打印测试结果总结
    print(f"\n{'='*55}")
    print(f"  TEST RESULTS ({len(errors)} samples)")
    print(f"{'='*55}")
    print(f"  Classification Accuracy : {test_acc:.1f}%")
    print(f"  MAE                     : {np.mean(errors):.2f} mm")
    print(f"  RMSE                    : {np.sqrt(np.mean(errors**2)):.2f} mm")
    print(f"  Max Error               : {np.max(errors):.2f} mm")
    print(f"  Avg Uncertainty (std)   : +/-{avg_std:.2f} mm")
    print(f"{'='*55}")

    # 逐样本明细
    for i in range(len(errors)):
        gt_r = REGION_NAMES[test_res['gt_cls'][i]]
        pd_r = REGION_NAMES[test_res['pred_cls'][i]]
        ok = "OK" if test_res['gt_cls'][i] == test_res['pred_cls'][i] else "X"
        gt_xy = f"({test_res['targets'][i,0]:6.1f},{test_res['targets'][i,1]:6.1f})"
        pd_xy = f"({test_res['preds'][i,0]:6.1f},{test_res['preds'][i,1]:6.1f})"
        print(f"  #{i} {gt_r}->{pd_r} [{ok}] GT{gt_xy} Pred{pd_xy} Err={errors[i]:.1f}mm")

    # 全景损伤定位散点图
    scatter_path = os.path.join(args.save_dir, 'test_scatter.png')
    plot_localization_scatter(
        test_res['targets'], test_res['preds'],
        log_vars=test_res['log_vars'],
        accuracy=test_acc, mean_std=avg_std,
        save_path=scatter_path, show=True)
    print(f"\n[SAVED] {scatter_path}")

    # Attention 拓扑图 (测试集样本的平均 attention)
    attn_path = os.path.join(args.save_dir, 'test_attention.png')
    mean_attn = np.mean(test_res['edge_attns'], axis=0)
    plot_attention_topology(
        mean_attn, model.gnn.edges,
        save_path=attn_path, show=True)
    print(f"[SAVED] {attn_path}")


if __name__ == '__main__':
    main()
