# -*- coding: utf-8 -*-
"""
model.py  —  小样本优化模型 (v4)
  - 轻量化 (~50K params, 适合 12 样本)
  - Inception 多尺度 + SE 保留
  - 移除几何补偿 (样本不足以学习稳定补偿)
  - Damage Index 融合分支
  - 直接回归坐标 (不依赖 softmax 加权中心)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import (
    NUM_REGIONS, EMBED_DIM, NUM_HEADS, IN_CHANNELS,
    REGION_CENTERS, normalize_coord, NUM_PAIRS,
)


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False), nn.ReLU(True),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _ = x.shape
        y = x.mean(dim=-1)
        return x * self.fc(y).unsqueeze(-1)


class InceptionStem(nn.Module):
    """轻量 Inception: 3/7/15 三尺度"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        bc = out_ch // 3
        extra = out_ch - 3 * bc
        self.b3  = nn.Sequential(nn.Conv1d(in_ch, bc, 3, padding=1), nn.BatchNorm1d(bc), nn.ReLU(True))
        self.b7  = nn.Sequential(nn.Conv1d(in_ch, bc, 7, padding=3), nn.BatchNorm1d(bc), nn.ReLU(True))
        self.b15 = nn.Sequential(nn.Conv1d(in_ch, bc + extra, 15, padding=7), nn.BatchNorm1d(bc + extra), nn.ReLU(True))
        self.se  = SEBlock(out_ch)
        self.pool = nn.MaxPool1d(4)

    def forward(self, x):
        out = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        return self.pool(self.se(out))


class LightEncoder(nn.Module):
    """轻量编码器: Inception → Conv+Pool × 2 → GAP"""
    def __init__(self, in_ch=IN_CHANNELS, dim=EMBED_DIM):
        super().__init__()
        self.inception = InceptionStem(in_ch, 32)       # → 32, L/4
        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(True), nn.MaxPool1d(4))
        self.conv2 = nn.Sequential(
            nn.Conv1d(48, dim, 3, padding=1), nn.BatchNorm1d(dim), nn.ReLU(True))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.inception(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x).squeeze(-1)


class NDTLocalizer(nn.Module):
    """
    v4 小样本优化模型

    架构:
      LightEncoder (每对66→embed_dim) 
        + Damage Index MLP 
        → 简单注意力加权池化
        → 分类头 (9类) + 回归头 (2D坐标)
        → 最终: region_center[argmax] + offset (不用softmax加权)
    """
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.encoder = LightEncoder(IN_CHANNELS, dim)

        # DI 分支: 把 66 个标量 DI 映射到向量
        self.di_branch = nn.Sequential(
            nn.Linear(NUM_PAIRS, 32), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(32, dim),
        )

        # 注意力权重 (per-pair importance)
        self.attn_score = nn.Sequential(
            nn.Linear(dim, 16), nn.Tanh(), nn.Linear(16, 1),
        )

        # 融合后预测
        fused_dim = dim * 2  # signal embed + DI embed
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 32), nn.ReLU(True),
            nn.Linear(32, NUM_REGIONS),
        )
        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 32), nn.ReLU(True),
            nn.Linear(32, 2), nn.Sigmoid(),  # 输出 [0,1] 归一化坐标
        )

        centers_norm = normalize_coord(REGION_CENTERS)
        self.register_buffer("region_centers", torch.from_numpy(centers_norm).float())

    def forward(self, x, di):
        """
        x:  (B, 66, 4, W)
        di: (B, 66) — Damage Index 标量
        """
        B, K, C, W = x.shape

        # 逐对编码
        emb = self.encoder(x.view(B * K, C, W)).view(B, K, -1)  # (B, 66, dim)

        # 注意力加权池化 (学习哪些传感器对最重要)
        attn_w = self.attn_score(emb).squeeze(-1)           # (B, 66)
        attn_w = F.softmax(attn_w, dim=-1)                   # (B, 66)
        signal_feat = torch.bmm(attn_w.unsqueeze(1), emb).squeeze(1)  # (B, dim)

        # DI 分支
        di_feat = self.di_branch(di)  # (B, dim)

        # 融合
        fused = torch.cat([signal_feat, di_feat], dim=-1)  # (B, dim*2)

        # 分类
        loc_logits = self.cls_head(fused)    # (B, 9)

        # 回归: 直接预测归一化坐标
        pred_coord = self.reg_head(fused)    # (B, 2) in [0,1]

        return {
            "loc_logits": loc_logits,
            "pred_loc": pred_coord,
            "attn_weights": attn_w,
        }


if __name__ == "__main__":
    model = NDTLocalizer()
    x = torch.randn(2, 66, IN_CHANNELS, 2048)
    di = torch.randn(2, 66)
    out = model(x, di)
    print(f"logits: {out['loc_logits'].shape}, pred: {out['pred_loc'].shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
