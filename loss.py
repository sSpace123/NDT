import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SENSOR_COORDS

# ========================================
# 五、 损失函数与柔性物理约束 
# ========================================

class PINNLoss(nn.Module):
    """
    自适应物理约束的多任务联合损失函数:
    L_total = L_cls + λ1 * L_reg + λ2 * L_phys
    """
    def __init__(self, lambda_reg=1.0, lambda_phys=0.1, num_classes=9, coords_min=-137.5, coords_max=137.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys
        
        # 平滑过拟合，提高鲁棒性
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.coords_min = coords_min
        self.coords_max = coords_max
        
        # 预计算 66 条边端点的二维物理坐标 (用于 L_phys计算)
        self.edges = []
        for i in range(12):
            for j in range(i+1, 12):
                self.edges.append((i, j))
        
        # 将传感器坐标转化为 Tensor
        self.sensors = torch.tensor(SENSOR_COORDS, dtype=torch.float32)

    def forward(self, reg_preds, cls_logits, edge_attns, targets_coords, targets_cls):
        """
        :param reg_preds: [B, 3] = [x_pred, y_pred, log_var]
        :param cls_logits: [B, 9] 分类输出
        :param edge_attns: [B, 66] 网络推断的边注意力分布
        :param targets_coords: [B, 2] 真实损伤坐标 (x, y)
        :param targets_cls: [B] 真实损伤区域标签 (0~8)
        """
        B = reg_preds.size(0)
        self.sensors = self.sensors.to(reg_preds.device)
        
        # ----------------------------------------------------
        # 1. L_cls: 区块分类的平滑交叉熵 
        # ----------------------------------------------------
        loss_cls = self.cls_loss(cls_logits, targets_cls)
        
        # ----------------------------------------------------
        # 2. L_reg: 带有异方差不确定性 (Aleatoric Uncertainty) 的回归损失
        # 解析预测的 x, y 及不确定度对数 log_sigma^2
        # ----------------------------------------------------
        preds_xy = reg_preds[:, :2] # [B, 2]
        log_var = reg_preds[:, 2:3] # [B, 1]
        
        # 均方误差 (MSE)
        mse_loss = F.mse_loss(preds_xy, targets_coords, reduction='none').mean(dim=1, keepdim=True) # [B, 1]
        
        # Uncertainty Loss: 0.5 * exp(-log_var) * MSE + 0.5 * log_var
        # 如果模型不确定，提升 log_var 从而降低第一项。正则项 0.5 * log_var 防止无限大
        loss_reg = 0.5 * torch.exp(-log_var) * mse_loss + 0.5 * log_var
        loss_reg = loss_reg.mean() # 标量
        
        # ----------------------------------------------------
        # 3. L_phys: 柔性物理约束 (Soft Time-of-Flight Constraint)
        # 用物理距离作为先验，引导 Attention 分布
        # 距离越小（通过该路径概率越大），我们期望得到的 Attention 也越大。
        # ----------------------------------------------------
        with torch.no_grad():
            # [B, 66] 保存理想的物理先验分布
            target_phys_attn = torch.zeros(B, len(self.edges), device=reg_preds.device)
            
            for idx, (u, v) in enumerate(self.edges):
                # 节点坐标
                pos_u = self.sensors[u] # [2]
                pos_v = self.sensors[v] # [2]
                
                # 网络预测的损伤位置，计算预测点到双传感器的总距离 D_ij
                # 如果要严格则用 targets_coords，但使用预测坐标可以作为自洽(Self-consistent)约束
                # 这里我们使用真实坐标来计算理想 Attention 分布作为一种监督信号 [B, 2]
                d1 = torch.norm(targets_coords - pos_u, dim=1) # [B]
                d2 = torch.norm(targets_coords - pos_v, dim=1) # [B]
                D_ij = d1 + d2 # 椭圆轨迹上的全飞行路径 [B]
                
                # 距离越短说明该路径捕捉到的波形越可能穿越损伤区域
                # (D_ij + eps) 取倒数
                target_phys_attn[:, idx] = 1.0 / (D_ij + 1e-8)
                
            # Softmax 将距离先验归一化分布
            target_phys_attn = F.softmax(target_phys_attn, dim=1) # [B, 66]
            
        # KL 散度约束模型提取的 attention (log_softmax) 接近物理分布
        log_edge_attns = torch.log(edge_attns + 1e-10) # 稳定对数
        # KLDivLoss 默认 expect input to be log-space and target to be linear space
        kl_loss = F.kl_div(log_edge_attns, target_phys_attn, reduction='batchmean')
        
        # ===== 最终多任务损失组合 =====
        total_loss = loss_cls + self.lambda_reg * loss_reg + self.lambda_phys * kl_loss
        
        return total_loss, {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_phys': kl_loss.item(),
            'mse': mse_loss.mean().item()
        }

if __name__ == '__main__':
    # 开发调试：验证 Loss 计算逻辑是否畅通
    device = torch.device('cpu')
    loss_fn = PINNLoss()
    reg_p = torch.randn(4, 3) # Batch=4, x,y,log_var
    cls_l = torch.randn(4, 9)
    attn = F.softmax(torch.randn(4, 66), dim=1)
    tgt_c = torch.randn(4, 2) * 100
    tgt_cls = torch.randint(0, 9, (4,))
    
    L_tot, dict_losses = loss_fn(reg_p, cls_l, attn, tgt_c, tgt_cls)
    print(f"Total Loss: {L_tot.item():.4f}")
    print(f"Loss Components: {dict_losses}")
