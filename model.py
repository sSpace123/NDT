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
    """2D CNN + SE: [B, 4, F, T] -> [B, edge_dim]
    输入 T=2048, 先用 AvgPool 压缩到 512, 大幅削减后续计算量"""
    def __init__(self, in_channels=IN_CHANNELS, out_channels=EDGE_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            # 时间维度早期降采样: [B, 4, 32, 2048] → [B, 4, 32, 512]
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
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
    36-edge 二分图: 左侧 0~5 -> 右侧 6~11.

    真正的动态注意力架构:
      1. prior_w: 物理距离倒数构成的可学习先验 [36]
      2. attn_mlp: 从 EdgeCNN 特征计算数据驱动 logits [B, 36, 1]
      3. 融合: (data_logits + prior_logits) → softmax → edge_attn [B, 36]
      4. 加权聚合: edge_attn * edge_feats → node 聚合
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

        # Step 1: 物理先验 — 距离倒数, 可学习
        init_logits = []
        for (i, j) in self.edges:
            dist = np.linalg.norm(SENSOR_COORDS[i] - SENSOR_COORDS[j])
            init_logits.append(1.0 / (dist + 1e-8))
        self.prior_w = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))

        # Step 2: 数据驱动注意力 MLP — 从边特征产生 logits
        self.attn_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim // 2, 1),
        )

        # 边特征投影到节点维度
        self.edge_to_node = nn.Linear(edge_dim, node_dim)

    def forward(self, edge_feats):
        """
        edge_feats: [B, 36, edge_dim]
        Returns:
            node_feats: [B, 12, node_dim]
            edge_attn:  [B, 36]  (经过 softmax 归一化的真实注意力)
        """
        B, E, D = edge_feats.size()

        # Step 2: 数据驱动 logits [B, 36, 1] → [B, 36]
        data_logits = self.attn_mlp(edge_feats).squeeze(-1)  # [B, 36]

        # Step 3: 融合 (加法) + Softmax → 真实 edge_attn
        prior_logits = self.prior_w.unsqueeze(0)  # [1, 36]
        edge_attn = F.softmax(data_logits + prior_logits, dim=1)  # [B, 36]

        # Step 4: 加权聚合 → 节点
        weighted = edge_feats * edge_attn.unsqueeze(-1)  # [B, 36, D]

        node_feats = torch.zeros(B, self.num_nodes, D, device=edge_feats.device)
        for idx, (u, v) in enumerate(self.edges):
            feat = weighted[:, idx, :]
            node_feats[:, u] = node_feats[:, u] + feat
            node_feats[:, v] = node_feats[:, v] + feat

        node_feats = F.relu(self.edge_to_node(node_feats))
        return node_feats, edge_attn


class PINNDamageLocator(nn.Module):
    """
    主模型: EdgeCNN -> BipartiteGNN (动态注意力) -> Pooling -> 分类 + 不确定性回归
    Input:  [B, 36, 4, F, T]
    Output: reg_out [B, 3], cls_logits [B, 9], edge_attn [B, 36]
    """
    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM,
                 node_dim=NODE_DIM, num_classes=NUM_CLASSES, num_nodes=NUM_NODES):
        super().__init__()
        self.edge_cnn = EdgeCNN(in_channels, edge_dim)
        self.gnn = BipartiteGNN(num_nodes, edge_dim, node_dim)

        self.cls_head = nn.Linear(node_dim, num_classes)
        self.reg_head = nn.Linear(node_dim, 3)  # [x, y, log_var]

    def forward(self, x):
        B, E, C, Fd, T = x.size()

        # Edge feature extraction
        edge_feats = self.edge_cnn(x.reshape(B * E, C, Fd, T)).view(B, E, -1)

        # Graph aggregation with dynamic attention
        node_feats, edge_attn = self.gnn(edge_feats)  # [B, 12, node_dim], [B, 36]

        # Global pooling: 均值聚合所有节点
        global_feat = node_feats.mean(dim=1)  # [B, node_dim]

        return self.reg_head(global_feat), self.cls_head(global_feat), edge_attn


if __name__ == '__main__':
    model = PINNDamageLocator()
    x = torch.randn(2, 36, 4, 32, 2048)
    reg, cls, attn = model(x)
    print(f"reg={reg.shape}, cls={cls.shape}, attn={attn.shape}")
    print(f"attn sum per sample: {attn.sum(dim=1)}")  # 应为 [1.0, 1.0]
