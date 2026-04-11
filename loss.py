import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LAMBDA_REG, LAMBDA_PHYS

# 不确定性 Warm-up 配置
WARMUP_EPOCHS = 999  # 强制全局纯 MSE 回归, 暂停不确定性估计


class PINNLoss(nn.Module):
    """
    多任务联合损失 (纯回归 + L2 多边协同正则化):
      L = L_cls + λ_reg * MSE + λ_phys * L2_penalty
    L2 正则化惩罚注意力稀疏性, 鼓励多边协同 (至少 3 条边参与定位).
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

        # 2. 回归损失 — 纯 MSE (WARMUP_EPOCHS=999, 全局禁用 log_var)
        preds_xy = reg_preds[:, :2]    # [B, 2]
        mse = F.mse_loss(preds_xy, targets_coords, reduction='none').mean(dim=1, keepdim=True)

        if epoch <= WARMUP_EPOCHS:
            # 纯 MSE, 不引入 log_var (梯度不会流向不确定性头)
            loss_reg = mse.mean()
            log_var_val = 0.0
        else:
            # 异方差不确定性回归 (保留但当前不可达)
            log_var = reg_preds[:, 2:3]
            log_var = torch.clamp(log_var, min=-4.0, max=4.0)
            loss_reg = (0.5 * torch.exp(-log_var) * mse + 0.5 * log_var).mean()
            log_var_val = log_var.mean().item()

        # 3. L2 多边协同正则化 (替代熵最小化)
        #    惩罚 ||attn||^2, 鼓励注意力分散到多条边, 而非坍缩到单一路径
        #    物理依据: 椭圆定位法至少需要 3 条边协同
        l2_penalty = (edge_attns ** 2).sum(dim=1).mean()

        total = loss_cls + self.lambda_reg * loss_reg + self.lambda_phys * l2_penalty

        return total, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_entropy': l2_penalty.item(),
            'mse': mse.mean().item(),
            'log_var': log_var_val,
        }
