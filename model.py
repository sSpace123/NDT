import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SENSOR_COORDS, NUM_NODES, EDGE_DIM, NODE_DIM, NUM_CLASSES, IN_CHANNELS
import numpy as np

# ========================================
# 四、 网络架构：自适应物理信息 GNN 
# ========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块，用于强化特定通道(可用于特征加权)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c) 
        y = F.relu(self.fc1(y))                    
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1) 
        return x * y.expand_as(x)

class EdgeCNN(nn.Module):
    """2D CNN + SE 提取各条边的特征 [B, 4, F, T] -> [B, 64]"""
    def __init__(self, in_channels=IN_CHANNELS, out_channels=EDGE_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=(1, 2)), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            SEBlock(16),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            SEBlock(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) 
        )
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv_layers(x) 
        x = x.view(x.size(0), -1) 
        edge_feat = F.relu(self.fc(x)) 
        return edge_feat

class SpatialGraphSubsystem(nn.Module):
    """
    负责基于先验距离初始化边权重，并随训练更新以自适应厚度和扭转变化。
    网络从 EdgeCNN 收取边特征，并聚合到节点上。
    """
    def __init__(self, num_nodes=NUM_NODES, edge_dim=EDGE_DIM, node_dim=NODE_DIM):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 二分图：左侧单列传感器 (0~5) 激发，右侧列传感器 (6~11) 接收，共 36 条边。
        self.edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if i < 6 and j >= 6:
                    self.edges.append((i, j))
                    
        self.num_edges = len(self.edges)
        
        # 计算 36 条边在二维坐标系下的欧氏距离倒数作为初始可学习权重
        init_weights = []
        coords = SENSOR_COORDS
        for (i, j) in self.edges:
            dist = np.linalg.norm(coords[i] - coords[j])
            init_weights.append(1.0 / (dist + 1e-8))
            
        # [36] 可学习参数，用物理倒数作先验约束初始化权重
        self.learnable_edge_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        
        # 节点聚合变换 (将各维度边特征及权重投影到Node维度)
        self.edge_to_node = nn.Linear(edge_dim, node_dim)
        
    def forward(self, edge_feats):
        # edge_feats: [B, E, edge_dim], E=36
        B, E, D = edge_feats.size()
        
        # 对边权重做 Softplus 以确保正则性或直接应用
        weights = F.softplus(self.learnable_edge_weights) # [E]
        weights = weights.view(1, E, 1) # [1, E, 1]
        
        # 融入自适应权重
        weighted_edge_feats = edge_feats * weights # [B, E, edge_dim]
        
        # Edge to Node 聚合机制 (Graph Conv)
        node_feats = torch.zeros(B, self.num_nodes, D, device=edge_feats.device)
        
        for idx, (u, v) in enumerate(self.edges):
            # 获取第 idx 条边的特征，并累加到对应的节点 [B, D]
            feat_idx = weighted_edge_feats[:, idx, :] 
            node_feats[:, u, :] = node_feats[:, u, :] + feat_idx
            node_feats[:, v, :] = node_feats[:, v, :] + feat_idx
            
        # [B, num_nodes, node_dim]
        node_feats = F.relu(self.edge_to_node(node_feats))
        return node_feats, weights.squeeze()

class PINNDamageLocator(nn.Module):
    """
    主模型架构：
    1. 接收 [B, 36, 4, F, 2048] 数据
    2. Edge CNN 提取路径特征
    3. GNN 聚合物理修正图逻辑
    4. Attention Pooling 生成注意力和路径贡献
    5. Uncertainty 回归头输出 [x, y, log_var] 和 九宫格分类结果
    """
    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM, node_dim=NODE_DIM, num_classes=NUM_CLASSES, num_nodes=NUM_NODES):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = 36
        
        self.edge_cnn = EdgeCNN(in_channels=in_channels, out_channels=edge_dim)
        self.spatial_gnn = SpatialGraphSubsystem(num_nodes=num_nodes, edge_dim=edge_dim, node_dim=node_dim)
        
        # Path-Level Attention Pooling 头
        self.attention_net = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.Tanh(),
            nn.Linear(node_dim // 2, 1, bias=False)
        )
        
        self.cls_head = nn.Linear(node_dim, num_classes)
        # 不确定性回归输出 -> [x_pred, y_pred, log_var]
        self.reg_head = nn.Linear(node_dim, 3) 
        
    def forward(self, x):
        # x: [B, 36, 4, F, T]
        B, E, C, F_dim, T = x.size()
        
        # == 1. Edge Feature Extraction ==
        x_reshaped = x.view(B * E, C, F_dim, T) # [B*36, 4, F, T]
        edge_feats = self.edge_cnn(x_reshaped)  # [B*36, 64]
        edge_feats = edge_feats.view(B, E, -1)  # [B, 36, 64]
        
        # == 2. Spatial Graph (Edge-to-Node) ==
        # node_feats: [B, 12, node_dim]
        node_feats, edge_weights = self.spatial_gnn(edge_feats)
        
        # == 3. Path-Level Attention Pooling ==
        # 计算节点到全局的 Attention [B, 12, 1]
        attn_scores = self.attention_net(node_feats)
        attn_weights = F.softmax(attn_scores, dim=1) # [B, 12, 1]
        
        # 重构回 36 条边的 Edge 级别 Attention 以用于熵正则化监督 [B, 36]
        edge_attn = torch.zeros(B, E, device=x.device)
        for idx, (u, v) in enumerate(self.spatial_gnn.edges):
            u_w = attn_weights[:, u, 0] # [B]
            v_w = attn_weights[:, v, 0] # [B]
            edge_attn[:, idx] = u_w * v_w
            
        edge_attn = edge_attn / (edge_attn.sum(dim=1, keepdim=True) + 1e-8) # [B, 36]
        
        # 全局图表示: Attention 聚合 Node
        global_feat = torch.sum(node_feats * attn_weights, dim=1) # [B, node_dim]
        
        # == 4. Output Heads ==
        cls_logits = self.cls_head(global_feat)
        reg_out = self.reg_head(global_feat) 
        
        return reg_out, cls_logits, edge_attn

if __name__ == '__main__':
    B, E, C, F_dim, T = 2, 36, 4, 32, 2048
    dummy_input = torch.randn(B, E, C, F_dim, T)
    model = PINNDamageLocator()
    out_reg, out_cls, out_attn = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Regression output shape: {out_reg.shape}")
    print(f"Classification logit shape: {out_cls.shape}")
    print(f"Edge Attention shape: {out_attn.shape}")
