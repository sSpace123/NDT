import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SENSOR_COORDS, NUM_NODES, EDGE_DIM, NODE_DIM, IN_CHANNELS


# ========================================
# 纯回归架构: EdgeCNN -> BipartiteGNN -> 坐标回归 (无分类头)
# ========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y)))).view(b, c, 1, 1)
        return x * y


class EdgeCNN(nn.Module):
    """2D CNN + SE: [B, 4, 32, 2048] -> [B, edge_dim]
    时间轴保护: 使用 AdaptiveMaxPool2d((1,8)) 保留 8 个时间窗口,
    维持声波到达时间差 (ToF) 和相位信息, 而非全局池化抹杀时序"""
    def __init__(self, in_channels=IN_CHANNELS, out_channels=EDGE_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, stride=(1, 2)),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            SEBlock(16),
            nn.Conv2d(16, 32, 3, padding=1, stride=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            SEBlock(32),
            nn.Conv2d(32, 64, 3, padding=1, stride=(1, 2)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 8)),  # 保留 8 个时间窗口
        )
        self.fc = nn.Linear(64 * 1 * 8, out_channels)  # 512 -> edge_dim

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return F.relu(self.fc(x))


class BipartiteGNN(nn.Module):
    """
    36-edge 二分图: 左侧 0~5 -> 右侧 6~11.
    物理先验 (距离倒数) + 数据驱动 MLP → softmax → 加权聚合
    """
    def __init__(self, num_nodes=NUM_NODES, edge_dim=EDGE_DIM, node_dim=NODE_DIM):
        super().__init__()
        self.num_nodes = num_nodes

        self.edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if i < 6 and j >= 6:
                    self.edges.append((i, j))
        self.num_edges = len(self.edges)

        # 物理先验: 距离倒数, 可学习
        init_logits = []
        for (i, j) in self.edges:
            dist = np.linalg.norm(SENSOR_COORDS[i] - SENSOR_COORDS[j])
            init_logits.append(1.0 / (dist + 1e-8))
        self.prior_w = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))

        # 数据驱动注意力 MLP
        self.attn_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim // 2, 1),
        )

        self.edge_to_node = nn.Linear(edge_dim, node_dim)

    def forward(self, edge_feats):
        B, E, D = edge_feats.size()

        data_logits = self.attn_mlp(edge_feats).squeeze(-1)
        prior_logits = self.prior_w.unsqueeze(0)
        edge_attn = F.softmax(data_logits + prior_logits, dim=1)

        weighted = edge_feats * edge_attn.unsqueeze(-1)

        node_feats = torch.zeros(B, self.num_nodes, D, device=edge_feats.device)
        for idx, (u, v) in enumerate(self.edges):
            feat = weighted[:, idx, :]
            node_feats[:, u] = node_feats[:, u] + feat
            node_feats[:, v] = node_feats[:, v] + feat

        node_feats = F.relu(self.edge_to_node(node_feats))
        return node_feats, edge_attn


class PINNDamageLocator(nn.Module):
    """
    纯回归模型: EdgeCNN -> BipartiteGNN -> 坐标回归
    Input:  [B, 36, 4, 32, 2048]
    Output: reg_out [B, 2] (归一化坐标), edge_attn [B, 36]
    注: 无分类头, 无 log_var, 纯粹连续空间坐标回归
    """
    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM,
                 node_dim=NODE_DIM, num_nodes=NUM_NODES):
        super().__init__()
        self.edge_cnn = EdgeCNN(in_channels, edge_dim)
        self.gnn = BipartiteGNN(num_nodes, edge_dim, node_dim)
        self.reg_head = nn.Linear(node_dim, 2)  # [x, y] 纯坐标回归

    def forward(self, x):
        B, E, C, Fd, T = x.size()
        edge_feats = self.edge_cnn(x.reshape(B * E, C, Fd, T)).view(B, E, -1)
        node_feats, edge_attn = self.gnn(edge_feats)
        global_feat = node_feats.mean(dim=1)
        return self.reg_head(global_feat), edge_attn


if __name__ == '__main__':
    model = PINNDamageLocator()
    x = torch.randn(2, 36, 4, 32, 2048)
    reg, attn = model(x)
    print(f"reg={reg.shape}, attn={attn.shape}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
