import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SENSOR_COORDS, NUM_NODES, EDGE_DIM, NODE_DIM, NUM_CLASSES, IN_CHANNELS


# ========================================
# 网络架构：自适应物理信息 GNN (36-Edge Bipartite)
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
    """2D CNN + SE: [B, 4, F, T] → [B, edge_dim]"""
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return F.relu(self.fc(x))


class BipartiteGNN(nn.Module):
    """
    36-edge 二分图: 左侧 0~5 → 右侧 6~11.
    物理距离倒数作为可学习边权初始值 (Soft Prior).
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
        self.num_edges = len(self.edges)

        # 物理距离倒数 → 可学习初始权重
        init_w = []
        for (i, j) in self.edges:
            dist = np.linalg.norm(SENSOR_COORDS[i] - SENSOR_COORDS[j])
            init_w.append(1.0 / (dist + 1e-8))
        self.edge_weights = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

        self.edge_to_node = nn.Linear(edge_dim, node_dim)

    def forward(self, edge_feats):
        """edge_feats: [B, 36, edge_dim] → node_feats: [B, 12, node_dim]"""
        B, E, D = edge_feats.size()

        w = F.softplus(self.edge_weights).view(1, E, 1)  # [1, 36, 1]
        weighted = edge_feats * w  # [B, 36, D]

        node_feats = torch.zeros(B, self.num_nodes, D, device=edge_feats.device)
        for idx, (u, v) in enumerate(self.edges):
            feat = weighted[:, idx, :]
            node_feats[:, u] = node_feats[:, u] + feat
            node_feats[:, v] = node_feats[:, v] + feat

        return F.relu(self.edge_to_node(node_feats))


class PINNDamageLocator(nn.Module):
    """
    主模型: EdgeCNN → BipartiteGNN → Attention Pooling → 分类 + 不确定性回归
    Input:  [B, 36, 4, F, T]
    Output: reg_out [B, 3], cls_logits [B, 9], edge_attn [B, 36]
    """
    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM,
                 node_dim=NODE_DIM, num_classes=NUM_CLASSES, num_nodes=NUM_NODES):
        super().__init__()
        self.edge_cnn = EdgeCNN(in_channels, edge_dim)
        self.gnn = BipartiteGNN(num_nodes, edge_dim, node_dim)

        self.attn_net = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.Tanh(),
            nn.Linear(node_dim // 2, 1, bias=False),
        )
        self.cls_head = nn.Linear(node_dim, num_classes)
        self.reg_head = nn.Linear(node_dim, 3)  # [x, y, log_var]

    def forward(self, x):
        B, E, C, Fd, T = x.size()

        # Edge feature extraction
        edge_feats = self.edge_cnn(x.reshape(B * E, C, Fd, T)).view(B, E, -1)

        # Graph aggregation
        node_feats = self.gnn(edge_feats)  # [B, 12, node_dim]

        # Node-level attention
        attn_scores = self.attn_net(node_feats)          # [B, 12, 1]
        attn_weights = F.softmax(attn_scores, dim=1)     # [B, 12, 1]

        # 重构 Edge-level attention (用于熵正则化)
        edge_attn = torch.zeros(B, E, device=x.device)
        for idx, (u, v) in enumerate(self.gnn.edges):
            edge_attn[:, idx] = attn_weights[:, u, 0] * attn_weights[:, v, 0]
        edge_attn = edge_attn / (edge_attn.sum(dim=1, keepdim=True) + 1e-8)

        # 全局特征
        global_feat = (node_feats * attn_weights).sum(dim=1)  # [B, node_dim]

        return self.reg_head(global_feat), self.cls_head(global_feat), edge_attn


if __name__ == '__main__':
    model = PINNDamageLocator()
    x = torch.randn(2, 36, 4, 32, 2048)
    reg, cls, attn = model(x)
    print(f"reg={reg.shape}, cls={cls.shape}, attn={attn.shape}")
