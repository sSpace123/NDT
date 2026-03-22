# -*- coding: utf-8 -*-
"""
config.py  —  配置 (v4 — 小样本优化)
"""
import os
import numpy as np

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(DATA_ROOT, "checkpoints")
PLOT_DIR  = os.path.join(DATA_ROOT, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SENSOR_COORDS = np.array([
    [-125.0,  125.0], [-125.0,   75.0], [-125.0,   25.0],
    [-125.0,  -25.0], [-125.0,  -75.0], [-125.0, -125.0],
    [ 125.0,  125.0], [ 125.0,   75.0], [ 125.0,   25.0],
    [ 125.0,  -25.0], [ 125.0,  -75.0], [ 125.0, -125.0],
], dtype=np.float32)

REGION_NAMES = ["左上", "上", "右上", "左中", "中", "右中", "左下", "下", "右下"]
REGION_DIRS  = ["左上", "上", "右上", "左中", "中", "右中", "左下", "下", "右下"]
REGION_CENTERS = np.array([
    [-75.0,  75.0], [0.0,  75.0], [75.0,  75.0],
    [-75.0,   0.0], [0.0,   0.0], [75.0,   0.0],
    [-75.0, -75.0], [0.0, -75.0], [75.0, -75.0],
], dtype=np.float32)
NUM_REGIONS = 9
NUM_PAIRS   = 66

SENSOR_PAIRS = []
for i in range(12):
    for j in range(i + 1, 12):
        SENSOR_PAIRS.append((i, j))
SENSOR_PAIRS = np.array(SENSOR_PAIRS, dtype=np.int64)

COORD_MIN   = -137.5
COORD_MAX   =  137.5
COORD_RANGE =  275.0

def normalize_coord(xy):
    return (np.asarray(xy, dtype=np.float32) - COORD_MIN) / COORD_RANGE
def denormalize_coord(xy_norm):
    return np.asarray(xy_norm, dtype=np.float32) * COORD_RANGE + COORD_MIN

CSV_HEADER_LINES = 21
WINDOW_LEN   = 2048      # 更长窗口, 保留更多信息
BATCH_SIZE   = 4          # 小 batch, 因为全 66 对
EPOCHS       = 300
LR           = 1e-3
WEIGHT_DECAY = 5e-3       # 强 L2 正则化 (小样本防过拟合)
LAMBDA_LOC   = 5.0        # 回归权重 (固定)
SEED         = 42

# 增强
AUGMENT_REPEAT = 100      # 更多重复
NOISE_STD      = 0.1
SCALE_RANGE    = (0.7, 1.3)
TIME_SHIFT_MAX = 500      # 最大时移

# 小波
WAVELET_NAME  = "db4"
WAVELET_LEVEL = 4
WAVELET_MODE  = "soft"

# 模型 (小型化)
IN_CHANNELS   = 4
EMBED_DIM     = 64        # 缩小 (原128)
NUM_HEADS     = 4
