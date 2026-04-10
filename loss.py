import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LAMBDA_REG, LAMBDA_PHYS


class PINNLoss(nn.Module):
    """
    多任务联合损失:
      L = L_cls + λ_reg * L_reg + λ_phys * H(attn)
    其中 H(attn) 为边注意力分布的信息熵, 优化器最小化 H 以鼓励聚焦决策.
    """

    def __init__(self, lambda_reg=LAMBDA_REG, lambda_phys=LAMBDA_PHYS):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, reg_preds, cls_logits, edge_attns, targets_coords, targets_cls):
        """
        reg_preds:     [B, 3]  (x, y, log_var) 归一化域 [-1, 1]
        cls_logits:    [B, 9]
        edge_attns:    [B, 36] 归一化注意力分布
        targets_coords:[B, 2]  归一化域 [-1, 1]
        targets_cls:   [B]
        """
        # 1. 分类损失
        loss_cls = self.cls_loss(cls_logits, targets_cls)

        # 2. 异方差不确定性回归损失
        preds_xy = reg_preds[:, :2]    # [B, 2]
        log_var = reg_preds[:, 2:3]    # [B, 1]
        log_var = torch.clamp(log_var, min=-6.0, max=6.0)  # 防止不确定性头失控
        mse = F.mse_loss(preds_xy, targets_coords, reduction='none').mean(dim=1, keepdim=True)
        loss_reg = (0.5 * torch.exp(-log_var) * mse + 0.5 * log_var).mean()

        # 3. 信息熵正则化 (最小化熵 → 鼓励聚焦)
        # H = -Σ p·log(p), H ≥ 0.  最小化 H 使分布更尖锐.
        entropy = -(edge_attns * torch.log(edge_attns + 1e-10)).sum(dim=1).mean()

        total = loss_cls + self.lambda_reg * loss_reg + self.lambda_phys * entropy

        return total, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_entropy': entropy.item(),
            'mse': mse.mean().item(),
        }
