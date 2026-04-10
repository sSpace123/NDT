import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================
# 五、 损失函数与柔性物理约束 
# ========================================

class PINNLoss(nn.Module):
    """
    自适应物理约束的多任务联合损失函数:
    L_total = L_cls + λ1 * L_reg + λ2 * L_phys (Entropy Minimization)
    """
    def __init__(self, lambda_reg=1.0, lambda_phys=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys
        
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, reg_preds, cls_logits, edge_attns, targets_coords, targets_cls):
        """
        :param reg_preds: [B, 3] = [x_pred, y_pred, log_var] 此时均在 [-1, 1] 归一化域
        :param cls_logits: [B, 9] 分类输出
        :param edge_attns: [B, 36] 网络推断的边注意力分布
        :param targets_coords: [B, 2] 真实损伤坐标 (x, y) 也是在 [-1, 1] 归一化域
        :param targets_cls: [B] 真实损伤区域标签 (0~8)
        """
        # 1. L_cls: 区块分类的平滑交叉熵 
        loss_cls = self.cls_loss(cls_logits, targets_cls)
        
        # 2. L_reg: 带有异方差不确定性的回归损失
        preds_xy = reg_preds[:, :2] # [B, 2]
        log_var = reg_preds[:, 2:3] # [B, 1]
        
        # 因为真实与预测的物理坐标都缩小到了 [-1, 1]，此处的 MSE 也收敛更好，避免了爆炸
        mse_loss = F.mse_loss(preds_xy, targets_coords, reduction='none').mean(dim=1, keepdim=True) # [B, 1]
        
        # Uncertainty Loss: 0.5 * exp(-log_var) * MSE + 0.5 * log_var
        loss_reg = 0.5 * torch.exp(-log_var) * mse_loss + 0.5 * log_var
        loss_reg = loss_reg.mean() # 标量
        
        # 3. L_phys: 信息熵正则化 (Entropy Regularization)
        # 鼓励网络分配注意力时“大胆”聚集于少量路径（低熵），而非平均分配。
        # 代替过去的僵化先验距离要求。物理距离的限制已通过网络内的权重初始化内化。
        loss_entropy = - torch.sum(edge_attns * torch.log(edge_attns + 1e-10), dim=1).mean()
        
        # ===== 最终多任务损失组合 =====
        total_loss = loss_cls + self.lambda_reg * loss_reg + self.lambda_phys * loss_entropy
        
        return total_loss, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_phys': loss_entropy.item(),
            'mse': mse_loss.mean().item()
        }

if __name__ == '__main__':
    # 开发调试：验证 Loss 计算逻辑是否畅通
    device = torch.device('cpu')
    loss_fn = PINNLoss()
    reg_p = torch.randn(4, 3) 
    cls_l = torch.randn(4, 9)
    attn = F.softmax(torch.randn(4, 36), dim=1)
    tgt_c = torch.randn(4, 2)
    tgt_cls = torch.randint(0, 9, (4,))
    
    L_tot, dict_losses = loss_fn(reg_p, cls_l, attn, tgt_c, tgt_cls)
    print(f"Total Loss: {L_tot.item():.4f}")
    print(f"Loss Components: {dict_losses}")
