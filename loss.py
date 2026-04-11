import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import (
    SENSOR_COORDS, LAMBDA_REG, LAMBDA_PHYS,
    DEAD_ZONE_R, COORD_RANGE, normalize_coord,
)


class GeometricPINNLoss(nn.Module):
    """带状容忍几何 PINN Loss:
      L = λ_reg · MSE(xy) + λ_phys · L_geom

    关键改进 (对比旧版 point-to-line):
      1. 线段距离: 计算点到**有限线段**的最短距离 (非无限延伸直线)
         若投影点在线段外则取到端点的距离, 物理上更正确
      2. 带状容忍 (Dead Zone): D_eff = max(0, D - R)
         导波散射区域有实际宽度 (约 ±10mm), 路径附近的预测不应受罚
         R 由 config.DEAD_ZONE_R 控制, 方便消融实验调参
    """

    def __init__(self, lambda_reg=LAMBDA_REG, lambda_phys=LAMBDA_PHYS):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_phys = lambda_phys

        # 预计算 36 条边的线段端点 (归一化坐标系 [-1, 1])
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if i < 6 and j >= 6:
                    edges.append((i, j))

        norm_sensors = np.array([normalize_coord(s) for s in SENSOR_COORDS])

        # 线段端点 A 和 B: [36, 2]
        seg_a = np.array([norm_sensors[i] for i, j in edges], dtype=np.float32)
        seg_b = np.array([norm_sensors[j] for i, j in edges], dtype=np.float32)

        self.register_buffer('seg_a', torch.from_numpy(seg_a))  # [36, 2]
        self.register_buffer('seg_b', torch.from_numpy(seg_b))  # [36, 2]

        # 将 DEAD_ZONE_R (mm) 转换到归一化空间 [-1, 1]
        # 归一化距离 = 物理距离 / (COORD_RANGE / 2)
        self.dead_zone_r = DEAD_ZONE_R / (COORD_RANGE / 2.0)

    def _point_to_segment_dist(self, px, py):
        """计算点 (px, py) 到 36 条线段的最短距离

        数学原理:
          线段 AB, 点 P. 将 AP 投影到 AB 方向: t = dot(AP, AB) / dot(AB, AB)
          - t ∈ [0,1]: 投影在线段内, 取垂直距离
          - t < 0: 投影在 A 外, 取 |PA|
          - t > 1: 投影在 B 外, 取 |PB|
          通过 clamp(t, 0, 1) 统一处理三种情况

        Args:
            px: [B, 1]  预测 x 坐标
            py: [B, 1]  预测 y 坐标
        Returns:
            dist: [B, 36]  到每条线段的最短距离
        """
        # 线段方向向量 AB
        abx = self.seg_b[:, 0] - self.seg_a[:, 0]  # [36]
        aby = self.seg_b[:, 1] - self.seg_a[:, 1]  # [36]

        # 向量 AP
        apx = px - self.seg_a[:, 0]  # [B, 36] 广播
        apy = py - self.seg_a[:, 1]  # [B, 36]

        # 投影参数 t = dot(AP, AB) / dot(AB, AB), clamp 到 [0, 1]
        ab_sq = abx ** 2 + aby ** 2 + 1e-10  # [36]
        t = (apx * abx + apy * aby) / ab_sq  # [B, 36]
        t = torch.clamp(t, 0.0, 1.0)         # [B, 36]

        # 线段上最近点 Q = A + t * AB
        qx = self.seg_a[:, 0] + t * abx  # [B, 36]
        qy = self.seg_a[:, 1] + t * aby  # [B, 36]

        # 欧氏距离 |PQ|
        dist = torch.sqrt((px - qx) ** 2 + (py - qy) ** 2 + 1e-10)  # [B, 36]
        return dist

    def forward(self, reg_preds, edge_attns, targets_coords):
        """
        reg_preds:      [B, 2]  归一化坐标 [-1, 1]
        edge_attns:     [B, 36] sigmoid 注意力
        targets_coords: [B, 2]  归一化坐标 [-1, 1]
        """
        # 1. MSE 回归损失
        mse = F.mse_loss(reg_preds, targets_coords)

        # 2. 带状容忍几何正则化
        px = reg_preds[:, 0:1]  # [B, 1]
        py = reg_preds[:, 1:2]  # [B, 1]

        # 点到线段距离
        dist = self._point_to_segment_dist(px, py)  # [B, 36]

        # 带状死区: D_eff = max(0, D - R), 路径 ±R 范围内不惩罚
        dist_eff = F.relu(dist - self.dead_zone_r)  # [B, 36]

        # L_geom = mean(attn_i × D_eff_i²)
        l_geom = (edge_attns * dist_eff ** 2).mean()

        total = self.lambda_reg * mse + self.lambda_phys * l_geom

        return total, {
            'mse': mse.item(),
            'l_geom': l_geom.item(),
        }


if __name__ == '__main__':
    loss_fn = GeometricPINNLoss()
    reg = torch.randn(2, 2) * 0.5
    attn = torch.sigmoid(torch.randn(2, 36))
    tgt = torch.randn(2, 2) * 0.5
    total, ld = loss_fn(reg, attn, tgt)
    print(f"total={total.item():.4f}, {ld}")
    print(f"seg_a={loss_fn.seg_a.shape}, dead_zone_r={loss_fn.dead_zone_r:.4f}")
    # 验证: 线段中点距离应为 0
    mid = (loss_fn.seg_a[0] + loss_fn.seg_b[0]) / 2
    _, ld2 = loss_fn(mid.unsqueeze(0), torch.ones(1, 36), mid.unsqueeze(0))
    print(f"midpoint sanity: l_geom={ld2['l_geom']:.6f} (应≈0)")
