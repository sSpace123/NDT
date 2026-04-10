"""
train.py — 完整训练/验证主入口脚本
========================================
用法:
    python train.py                          # 使用默认参数训练
    python train.py --epochs 200 --lr 1e-3   # 自定义超参
    python train.py --demo                   # 用合成数据快速验证流水线
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from config import (
    SENSOR_COORDS, REGION_NAMES, COORD_MIN, COORD_MAX,
    FS, FC, CG, WINDOW_SIZE, EPSILON
)
from model import PINNDamageLocator
from loss import PINNLoss
from vis import plot_training_curves, plot_localization_scatter, plot_attention_topology


# ========================================
# 合成演示数据集 (用于验证整个 Pipeline)
# ========================================
class SyntheticSHMDataset(Dataset):
    """
    生成合成随机数据，用于验证模型维度与训练流程是否正确运行。
    每条样本: (data [66, 4, 32, 2048], coord [2], cls [scalar])
    """
    def __init__(self, num_samples=64, num_edges=66, channels=4, freq_bins=32, time_steps=2048):
        super().__init__()
        self.num_samples = num_samples
        self.num_edges = num_edges
        self.channels = channels
        self.freq_bins = freq_bins
        self.time_steps = time_steps

        # 预生成随机标签
        self.coords = np.random.uniform(COORD_MIN, COORD_MAX, (num_samples, 2)).astype(np.float32)
        self.cls_labels = np.random.randint(0, 9, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机特征张量 [66, 4, 32, 2048]
        data = np.random.randn(
            self.num_edges, self.channels, self.freq_bins, self.time_steps
        ).astype(np.float32) * 0.1

        return (
            torch.from_numpy(data),
            torch.from_numpy(self.coords[idx]),
            torch.tensor(self.cls_labels[idx], dtype=torch.long)
        )


# ========================================
# 训练核心逻辑
# ========================================
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """运行一个 epoch 的训练"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    for batch_idx, (data, coords, cls_labels) in enumerate(dataloader):
        # data: [B, 66, 4, F, T] | coords: [B, 2] | cls_labels: [B]
        data = data.to(device)
        coords = coords.to(device)
        cls_labels = cls_labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        reg_out, cls_logits, edge_attn = model(data)

        # 计算多任务联合损失
        loss, loss_dict = loss_fn(reg_out, cls_logits, edge_attn, coords, cls_labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止导波物理约束导致的梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss_dict['loss_cls'] + loss_dict['loss_reg'] + loss_dict['loss_phys']
        total_mse += loss_dict['mse']
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Cls: {loss_dict['loss_cls']:.4f} | "
                  f"Reg: {loss_dict['loss_reg']:.4f} | "
                  f"Phys: {loss_dict['loss_phys']:.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = np.sqrt(total_mse / max(num_batches, 1))
    return avg_loss, avg_rmse


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """运行验证集评估"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []
    all_log_vars = []
    all_edge_attns = []

    for data, coords, cls_labels in dataloader:
        data = data.to(device)
        coords = coords.to(device)
        cls_labels = cls_labels.to(device)

        reg_out, cls_logits, edge_attn = model(data)
        loss, loss_dict = loss_fn(reg_out, cls_logits, edge_attn, coords, cls_labels)

        total_loss += loss_dict['loss_cls'] + loss_dict['loss_reg'] + loss_dict['loss_phys']
        total_mse += loss_dict['mse']
        num_batches += 1

        # 收集预测结果用于可视化
        all_preds.append(reg_out[:, :2].cpu().numpy())
        all_targets.append(coords.cpu().numpy())
        all_log_vars.append(reg_out[:, 2].cpu().numpy())
        all_edge_attns.append(edge_attn.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = np.sqrt(total_mse / max(num_batches, 1))

    results = {
        'preds': np.concatenate(all_preds, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'log_vars': np.concatenate(all_log_vars, axis=0),
        'edge_attns': np.concatenate(all_edge_attns, axis=0),
    }

    return avg_loss, avg_rmse, results


def main():
    parser = argparse.ArgumentParser(description='GNN-SHM 损伤定位系统训练脚本')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='回归损失权重 λ1')
    parser.add_argument('--lambda_phys', type=float, default=0.1, help='物理约束权重 λ2')
    parser.add_argument('--demo', action='store_true', help='使用合成数据验证流水线')
    parser.add_argument('--save_dir', type=str, default='outputs', help='输出保存目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    args = parser.parse_args()

    # ---------- 设备选择 ----------
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # ---------- 输出目录 ----------
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- 数据集 ----------
    if args.demo:
        print("=" * 60)
        print("  [DEMO 模式] 使用合成随机数据验证训练流水线")
        print("=" * 60)
        full_dataset = SyntheticSHMDataset(num_samples=32)
    else:
        # 用户自定义: 在此处替换为真实 SHMGraphDataset
        # from dataset import SHMGraphDataset
        # full_dataset = SHMGraphDataset(sample_list=your_sample_list, augment=True)
        print("请使用 --demo 标志进行流水线验证，或替换此处为真实数据集。")
        print("示例: python train.py --demo")
        sys.exit(0)

    # 计算训练/验证划分比例（80/20）
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"训练集: {n_train} 样本 | 验证集: {n_val} 样本")

    # ---------- 模型、损失、优化器 ----------
    model = PINNDamageLocator(
        in_channels=4,
        edge_dim=64,
        node_dim=128,
        num_classes=9,
        num_nodes=12
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    loss_fn = PINNLoss(
        lambda_reg=args.lambda_reg,
        lambda_phys=args.lambda_phys
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---------- 训练循环 ----------
    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []
    best_val_rmse = float('inf')

    print(f"\n开始训练 ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 训练
        t_loss, t_rmse = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        train_losses.append(t_loss)
        train_rmses.append(t_rmse)

        # 验证
        v_loss, v_rmse, val_results = validate(model, val_loader, loss_fn, device)
        val_losses.append(v_loss)
        val_rmses.append(v_rmse)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {t_loss:.4f} RMSE: {t_rmse:.2f}mm | "
              f"Val Loss: {v_loss:.4f} RMSE: {v_rmse:.2f}mm | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.1f}s")

        # 保存最优模型
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

    # ---------- 可视化 ----------
    print("生成训练可视化图表...")

    # 1. 训练曲线
    plot_training_curves(
        train_losses, val_losses, train_rmses, val_rmses,
        save_path=os.path.join(args.save_dir, 'training_curves.png')
    )

    # 2. 定位散点图 (使用最后一个 epoch 的验证结果)
    if val_results['preds'].shape[0] > 0:
        plot_localization_scatter(
            val_results['targets'],
            val_results['preds'],
            val_results['log_vars'],
            save_path=os.path.join(args.save_dir, 'localization_scatter.png')
        )

    # 3. Attention 拓扑图 (取验证集第一条样本的边注意力)
    if val_results['edge_attns'].shape[0] > 0:
        plot_attention_topology(
            val_results['edge_attns'][0],
            save_path=os.path.join(args.save_dir, 'attention_topology.png')
        )

    print(f"所有输出已保存至: {os.path.abspath(args.save_dir)}")


if __name__ == '__main__':
    main()
