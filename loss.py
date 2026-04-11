import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LAMBDA_REG, LAMBDA_PHYS


class PINNLoss(nn.Module):
    """
    纯回归损失 + L2 多边协同正则化:
      L = λ_reg * MSE(xy) + λ_phys * ||attn||²
    
    物理依据:
    - 纯 MSE 回归: 无 log_var, 无分类交叉熵, 避免小样本下梯度冲突
    - L2 attention 正则: 惩罚 ||attn||², 鼓励多边协同 (椭圆定位至少需 3 条路径)
      而非最小化熵 (最小化熵→单边坍缩→物理不合理)
    """

    def __init__(self, lambda_reg=LAMBDA_REG, lambda_phys=LAMBDA_PHYS):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys

    def forward(self, reg_preds, edge_attns, targets_coords):
        """
        reg_preds:     [B, 2]  归一化域 [-1, 1] 坐标
        edge_attns:    [B, 36] softmax 归一化注意力
        targets_coords:[B, 2]  归一化域 [-1, 1]
        """
        # 1. MSE 回归损失
        mse = F.mse_loss(reg_preds, targets_coords)

        # 2. L2 多边协同正则化: 惩罚稀疏注意力
        l2_penalty = (edge_attns ** 2).sum(dim=1).mean()

        total = self.lambda_reg * mse + self.lambda_phys * l2_penalty

        return total, {
            'loss_total': total.item(),
            'mse': mse.item(),
            'l2_attn': l2_penalty.item(),
        }
