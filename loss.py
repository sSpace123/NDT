import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LAMBDA_REG, LAMBDA_PHYS

# 不确定性 Warm-up 配置
WARMUP_EPOCHS = 15  # 前 15 个 epoch 禁用 log_var 学习, 纯 MSE 优化


class PINNLoss(nn.Module):
    """
    多任务联合损失 (带 Warm-up):
      Phase 1 (epoch ≤ WARMUP_EPOCHS): L = L_cls + λ_reg * MSE  (纯回归, 无不确定性)
      Phase 2 (epoch > WARMUP_EPOCHS): L = L_cls + λ_reg * L_hetero + λ_phys * H(attn)
    """

    def __init__(self, lambda_reg=LAMBDA_REG, lambda_phys=LAMBDA_PHYS):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, reg_preds, cls_logits, edge_attns,
                targets_coords, targets_cls, epoch=999):
        """
        reg_preds:     [B, 3]  (x, y, log_var) 归一化域 [-1, 1]
        cls_logits:    [B, 9]
        edge_attns:    [B, 36] 归一化注意力分布
        targets_coords:[B, 2]  归一化域 [-1, 1]
        targets_cls:   [B]
        epoch:         当前 epoch (1-indexed), 用于 warm-up 控制
        """
        # 1. 分类损失
        loss_cls = self.cls_loss(cls_logits, targets_cls)

        # 2. 回归损失 — 分阶段
        preds_xy = reg_preds[:, :2]    # [B, 2]
        mse = F.mse_loss(preds_xy, targets_coords, reduction='none').mean(dim=1, keepdim=True)

        if epoch <= WARMUP_EPOCHS:
            # Phase 1: 纯 MSE, 不引入 log_var (梯度不会流向不确定性头)
            loss_reg = mse.mean()
            log_var_val = 0.0
        else:
            # Phase 2: 异方差不确定性回归
            log_var = reg_preds[:, 2:3]                      # [B, 1]
            log_var = torch.clamp(log_var, min=-4.0, max=4.0) # 收紧: exp(4)≈55x, exp(-4)≈0.02x
            loss_reg = (0.5 * torch.exp(-log_var) * mse + 0.5 * log_var).mean()
            log_var_val = log_var.mean().item()

        # 3. 信息熵正则化 (最小化熵 → 鼓励聚焦)
        entropy = -(edge_attns * torch.log(edge_attns + 1e-10)).sum(dim=1).mean()

        total = loss_cls + self.lambda_reg * loss_reg + self.lambda_phys * entropy

        return total, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_entropy': entropy.item(),
            'mse': mse.mean().item(),
            'log_var': log_var_val,
        }
