import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import (
    SENSOR_COORDS, NUM_NODES, EDGE_DIM, NODE_DIM, IN_CHANNELS,
    TIME_REDUCED_LEN,
)

# 推导 1D 时间池化的 kernel_size: 2048 / TIME_REDUCED_LEN
_INPUT_TIME_LEN = 2048
_TIME_POOL_K = _INPUT_TIME_LEN // TIME_REDUCED_LEN  # 默认 2048/128 = 16


class EdgeCNN(nn.Module):
    """时序保护的边特征提取器: [B, 4, 32, 2048] -> [B, edge_dim]

    Step 1: 2D Conv 逐步压缩频率维度 (32→1), 保留完整时间轴
    Step 2: 1D AvgPool 平滑降维时间轴 (2048→TIME_REDUCED_LEN), 保留 ToF
    Step 3: GAP + Linear 输出 edge_dim
    """
    def __init__(self, in_channels=IN_CHANNELS, out_channels=EDGE_DIM):
        super().__init__()
        # 2D Conv: 压缩频率, 保持时间不变
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, stride=(2, 1)),  # freq: 32→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 1)),          # freq: 16→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (8, 1)),                                # freq: 8→1
            nn.ReLU(),
        )
        # 1D 时间轴降维: 2048 → TIME_REDUCED_LEN (默认 128)
        self.time_pool = nn.AvgPool1d(kernel_size=_TIME_POOL_K,
                                       stride=_TIME_POOL_K)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv2d(x)              # [B, 64, 1, 2048]
        x = x.view(x.size(0), 64, -1)  # [B, 64, 2048] 防御性 flatten
        x = self.time_pool(x)           # [B, 64, TIME_REDUCED_LEN]
        x = x.mean(dim=2)               # [B, 64] GAP
        return F.relu(self.fc(x))        # [B, edge_dim]


class BipartiteGNN(nn.Module):
    """36-edge 二分图, sigmoid 独立注意力

    - sigmoid: 每条边独立 [0,1], 支持多边同时高权重
    - prior_w: 距离倒数偏置 (可学习)
    - 单层 Linear 注意力 (无 MLP)
    """
    def __init__(self, num_nodes=NUM_NODES, edge_dim=EDGE_DIM, node_dim=NODE_DIM):
        super().__init__()
        self.num_nodes = num_nodes

        self.edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if i < 6 and j >= 6:
                    self.edges.append((i, j))
        assert len(self.edges) == 36

        # 物理先验偏置
        init_bias = []
        for (i, j) in self.edges:
            dist = np.linalg.norm(SENSOR_COORDS[i] - SENSOR_COORDS[j])
            init_bias.append(1.0 / (dist + 1e-8))
        self.prior_w = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

        self.attn_linear = nn.Linear(edge_dim, 1)
        self.edge_to_node = nn.Linear(edge_dim, node_dim)

    def forward(self, edge_feats):
        B, E, D = edge_feats.size()
        logits = self.attn_linear(edge_feats).squeeze(-1) + self.prior_w
        edge_attn = torch.sigmoid(logits)

        weighted = edge_feats * edge_attn.unsqueeze(-1)
        node_feats = torch.zeros(B, self.num_nodes, D, device=edge_feats.device)
        for idx, (u, v) in enumerate(self.edges):
            feat = weighted[:, idx, :]
            node_feats[:, u] = node_feats[:, u] + feat
            node_feats[:, v] = node_feats[:, v] + feat

        return F.relu(self.edge_to_node(node_feats)), edge_attn


class PINNDamageLocator(nn.Module):
    """纯回归: EdgeCNN → BipartiteGNN → [B, 2] 坐标"""

    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM,
                 node_dim=NODE_DIM, num_nodes=NUM_NODES):
        super().__init__()
        self.edge_cnn = EdgeCNN(in_channels, edge_dim)
        self.gnn = BipartiteGNN(num_nodes, edge_dim, node_dim)
        self.reg_head = nn.Linear(node_dim, 2)

    def forward(self, x):
        B, E, C, Fd, T = x.size()
        assert E == 36 and C == IN_CHANNELS
        edge_feats = self.edge_cnn(x.reshape(B * E, C, Fd, T)).view(B, E, -1)
        node_feats, edge_attn = self.gnn(edge_feats)
        return self.reg_head(node_feats.mean(dim=1)), edge_attn


if __name__ == '__main__':
    model = PINNDamageLocator()
    x = torch.randn(2, 36, 4, 32, 2048)
    reg, attn = model(x)
    print(f"reg={reg.shape}, attn={attn.shape}")
    print(f"attn range: [{attn.min():.3f}, {attn.max():.3f}]")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
