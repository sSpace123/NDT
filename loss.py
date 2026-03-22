# -*- coding: utf-8 -*-
"""
loss.py  —  固定权重损失函数 (v4)
  - 固定 cls/reg 权重 (避免不稳定的自适应)
  - 区域边界约束保留
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import NUM_REGIONS, REGION_CENTERS, normalize_coord, COORD_RANGE, LAMBDA_LOC


class NDTLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, lambda_loc=LAMBDA_LOC):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.reg_loss = nn.SmoothL1Loss()
        self.lambda_loc = lambda_loc

        centers_norm = normalize_coord(REGION_CENTERS)
        self.register_buffer("centers", torch.from_numpy(centers_norm).float())
        self.boundary_r = 37.5 / COORD_RANGE

    def forward(self, model_output, region_idx, center_norm):
        logits = model_output["loc_logits"]
        pred = model_output["pred_loc"]

        loss_cls = self.cls_loss(logits, region_idx)
        loss_reg = self.reg_loss(pred, center_norm)

        # 边界约束
        pred_cls = logits.argmax(dim=-1)
        pred_centers = self.centers[pred_cls]
        dist = torch.norm(pred - pred_centers, dim=-1)
        violation = F.relu(dist - self.boundary_r)
        loss_bnd = violation.mean()

        total = loss_cls + self.lambda_loc * loss_reg + loss_bnd

        return {
            "total": total,
            "cls": loss_cls.detach(),
            "reg": loss_reg.detach(),
            "bnd": loss_bnd.detach(),
        }
