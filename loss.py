import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SENSOR_COORDS, LAMBDA_REG, LAMBDA_PHYS, normalize_coord


class PINNLoss(nn.Module):
    """几何一致性 PINN Loss (纯回归):
      L = λ_reg · MSE(xy) + λ_phys · L_geom

    L_geom 的物理含义 (延时叠加思想):
      每条边 (i,j) 对应两个传感器之间的直射路径 (直线).
      损伤点若确实在某条路径上, 该路径的散射信号最强 → attention 应该高.
      L_geom = mean( attn_i × d_i² )  强制预测点落在高注意力路径的交汇处.
      这比 L2-penalty/entropy 更有物理意义: 直接将几何约束编码进损失.
    """

    def __init__(self, lambda_reg=LAMBDA_REG, lambda_phys=LAMBDA_PHYS):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys

        # 预计算 36 条边的直线方程系数 (在归一化坐标系 [-1,1] 下)
        # 直线 ax + by + c = 0, 预归一化使 sqrt(a²+b²)=1, 则点到线距离 = |ax+by+c|
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if i < 6 and j >= 6:
                    edges.append((i, j))

        # 将传感器坐标归一化到 [-1, 1], 与模型预测同空间
        norm_sensors = np.array([normalize_coord(s) for s in SENSOR_COORDS])

        line_coeffs = []
        for (i, j) in edges:
            x1, y1 = norm_sensors[i]
            x2, y2 = norm_sensors[j]
            a = float(y2 - y1)
            b = float(x1 - x2)
            c = float(x2 * y1 - x1 * y2)
            norm = np.sqrt(a ** 2 + b ** 2) + 1e-10
            line_coeffs.append([a / norm, b / norm, c / norm])  # 预归一化

        # register_buffer: 随模型移动到 GPU, 但不参与梯度
        self.register_buffer(
            'line_coeffs',
            torch.tensor(line_coeffs, dtype=torch.float32))  # [36, 3]

    def forward(self, reg_preds, edge_attns, targets_coords):
        """
        reg_preds:      [B, 2]  归一化坐标 [-1, 1]
        edge_attns:     [B, 36] sigmoid 注意力 (独立, 非 softmax)
        targets_coords: [B, 2]  归一化坐标 [-1, 1]
        """
        # 1. 坐标 MSE 回归损失
        mse = F.mse_loss(reg_preds, targets_coords)

        # 2. 几何 PINN 正则化: 预测点到 36 条传感器直线的加权距离
        xp = reg_preds[:, 0:1]  # [B, 1]
        yp = reg_preds[:, 1:2]  # [B, 1]

        a = self.line_coeffs[:, 0]  # [36]
        b = self.line_coeffs[:, 1]  # [36]
        c = self.line_coeffs[:, 2]  # [36]

        # 点到直线距离 (已预归一化, |ax+by+c| 即为真实距离)
        dist = torch.abs(xp * a + yp * b + c)  # [B, 36]

        # L_geom = mean( attn_i × d_i² ): 高注意力的路径必须穿过预测点
        l_geom = (edge_attns * dist ** 2).mean()

        total = self.lambda_reg * mse + self.lambda_phys * l_geom

        return total, {
            'mse': mse.item(),
            'l_geom': l_geom.item(),
        }


if __name__ == '__main__':
    loss_fn = PINNLoss()
    reg = torch.randn(2, 2) * 0.5
    attn = torch.sigmoid(torch.randn(2, 36))
    tgt = torch.randn(2, 2) * 0.5
    total, ld = loss_fn(reg, attn, tgt)
    print(f"total={total.item():.4f}, {ld}")
    print(f"line_coeffs shape: {loss_fn.line_coeffs.shape}")
