import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SENSOR_COORDS, NUM_NODES, EDGE_DIM, NODE_DIM, IN_CHANNELS


class EdgeCNN(nn.Module):
    """时序保护的边特征提取器: [B, 4, 32, 2048] -> [B, edge_dim]

    架构设计 (对比旧版 AdaptiveAvgPool2d 的改进):
      Step 1: 2D Conv 压缩频率维度 (32→1), 提取通道-频率交叉特征
      Step 2: 1D AvgPool 在时间轴平滑降维 (2048→128), 保留相对飞行时间 (ToF)
      Step 3: GAP 全局平均池化 + Linear 输出 edge_dim
    """
    def __init__(self, in_channels=IN_CHANNELS, out_channels=EDGE_DIM):
        super().__init__()
        # Step 1: 2D Conv 逐步压缩频率维度, 保留时间轴
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, stride=(2, 1)),  # freq: 32→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 1)),          # freq: 16→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (8, 1)),                                # freq: 8→1, 彻底折叠频率
            nn.ReLU(),
        )
        # Step 2: 1D AvgPool 平滑降维时间轴, 保留 ToF 相对结构
        self.time_pool = nn.AvgPool1d(kernel_size=16, stride=16)     # 2048→128
        # Step 3: GAP + Linear
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        B = x.size(0)
        x = self.conv2d(x)              # [B, 64, 1, 2048]
        assert x.size(2) == 1, f"频率维未折叠: {x.shape}"
        x = x.squeeze(2)                # [B, 64, 2048]
        x = self.time_pool(x)           # [B, 64, 128] 保留 ToF 时序结构
        x = x.mean(dim=2)               # [B, 64] GAP
        return F.relu(self.fc(x))        # [B, edge_dim]


class BipartiteGNN(nn.Module):
    """36-edge 二分图: 左侧 0~5 → 右侧 6~11

    注意力设计 (对比旧版 softmax 的改进):
      - sigmoid 独立激活: 每条边独立 [0,1], 支持多边同时高权重 (物理协同)
      - softmax 会强制 Σ=1, 导致边之间零和竞争, 与多路径定位矛盾
      - 距离倒数先验 prior_w 作为简单偏置, 无需 MLP
    """
    def __init__(self, num_nodes=NUM_NODES, edge_dim=EDGE_DIM, node_dim=NODE_DIM):
        super().__init__()
        self.num_nodes = num_nodes

        # 构建二分图边表
        self.edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if i < 6 and j >= 6:
                    self.edges.append((i, j))
        assert len(self.edges) == 36

        # 物理先验: 距离倒数偏置 (可学习)
        init_bias = []
        for (i, j) in self.edges:
            dist = np.linalg.norm(SENSOR_COORDS[i] - SENSOR_COORDS[j])
            init_bias.append(1.0 / (dist + 1e-8))
        self.prior_w = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

        # 极简注意力: 单层 Linear (无 MLP)
        self.attn_linear = nn.Linear(edge_dim, 1)

        # 边→节点投影
        self.edge_to_node = nn.Linear(edge_dim, node_dim)

    def forward(self, edge_feats):
        """
        edge_feats: [B, 36, edge_dim]
        Returns: node_feats [B, 12, node_dim], edge_attn [B, 36]
        """
        B, E, D = edge_feats.size()

        # 独立 sigmoid 注意力: 数据 logit + 物理偏置
        logits = self.attn_linear(edge_feats).squeeze(-1) + self.prior_w  # [B, 36]
        edge_attn = torch.sigmoid(logits)  # [B, 36] 每条边独立 0~1

        # 加权聚合到节点
        weighted = edge_feats * edge_attn.unsqueeze(-1)  # [B, 36, D]

        node_feats = torch.zeros(B, self.num_nodes, D, device=edge_feats.device)
        for idx, (u, v) in enumerate(self.edges):
            feat = weighted[:, idx, :]
            node_feats[:, u] = node_feats[:, u] + feat
            node_feats[:, v] = node_feats[:, v] + feat

        node_feats = F.relu(self.edge_to_node(node_feats))
        return node_feats, edge_attn


class PINNDamageLocator(nn.Module):
    """纯回归定位模型: EdgeCNN → BipartiteGNN → 坐标回归
    Output: reg_out [B, 2], edge_attn [B, 36]"""

    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM,
                 node_dim=NODE_DIM, num_nodes=NUM_NODES):
        super().__init__()
        self.edge_cnn = EdgeCNN(in_channels, edge_dim)
        self.gnn = BipartiteGNN(num_nodes, edge_dim, node_dim)
        self.reg_head = nn.Linear(node_dim, 2)

    def forward(self, x):
        B, E, C, F, T = x.size()
        assert E == 36 and C == IN_CHANNELS
        edge_feats = self.edge_cnn(x.reshape(B * E, C, F, T)).view(B, E, -1)
        node_feats, edge_attn = self.gnn(edge_feats)
        global_feat = node_feats.mean(dim=1)
        return self.reg_head(global_feat), edge_attn


if __name__ == '__main__':
    model = PINNDamageLocator()
    x = torch.randn(2, 36, 4, 32, 2048)
    reg, attn = model(x)
    print(f"reg={reg.shape}, attn={attn.shape}")
    print(f"attn range: [{attn.min():.3f}, {attn.max():.3f}]")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
