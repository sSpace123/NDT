import os
import numpy as np

# ========================================
# 1. 物理参数与传感器配置
# ========================================

FS = 1e7          # 采样率 (Hz), 10 MHz
FC = 100e3        # 激励信号中心频率 (Hz), 100 kHz 五周期 Tone burst

# 传感器坐标 (mm): 左侧 0~5, 右侧 6~11 (用于构建二分图)
SENSOR_COORDS = np.array([
    [-125.0,  125.0], [-125.0,   75.0], [-125.0,   25.0],
    [-125.0,  -25.0], [-125.0,  -75.0], [-125.0, -125.0],
    [ 125.0,  125.0], [ 125.0,   75.0], [ 125.0,   25.0],
    [ 125.0,  -25.0], [ 125.0,  -75.0], [ 125.0, -125.0],
], dtype=np.float32)

NUM_NODES = 12
NUM_CLASSES = 9

REGION_NAMES = ["左上", "上", "右上", "左中", "中", "右中", "左下", "下", "右下"]

# 区域中心地标 (mm), 用于粗分类标签
REGION_CENTERS = np.array([
    [-83.3,  83.3], [  0.0,  83.3], [ 83.3,  83.3],
    [-83.3,   0.0], [  0.0,   0.0], [ 83.3,   0.0],
    [-83.3, -83.3], [  0.0, -83.3], [ 83.3, -83.3],
], dtype=np.float32)

# 结构坐标极值 (mm)
COORD_MIN = -137.5
COORD_MAX = 137.5
COORD_RANGE = COORD_MAX - COORD_MIN  # 275.0


def normalize_coord(coord):
    """物理坐标 [-137.5, 137.5] → 归一化 [-1, 1]"""
    return (np.array(coord, dtype=np.float32) - COORD_MIN) / COORD_RANGE * 2.0 - 1.0


def denormalize_coord(norm_coord):
    """归一化 [-1, 1] → 物理坐标 [-137.5, 137.5]"""
    return (np.array(norm_coord, dtype=np.float32) + 1.0) / 2.0 * COORD_RANGE + COORD_MIN


# ========================================
# 2. 数据处理配置 — 动态绝对路径
# ========================================

# config.py 所在目录 = gnn_shm/，其父目录 = 模拟损伤/ (包含中文子文件夹)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(_THIS_DIR)

# 区域子目录名: 与 REGION_NAMES 索引一一对应
REGION_DIRS = ["左上", "上", "右上", "左中", "中", "右中", "左下", "下", "右下"]

WINDOW_HALF_SIZE = 1024   # 中心峰值两侧各截取点数
NUM_PAIRS = 66            # CSV 采集总文件对数 (C(12,2))
CSV_HEADER_LINES = 21     # 示波器头文件行数

# CWT 特征通道数
IN_CHANNELS = 4

# 小波去噪参数
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 4

# ========================================
# 3. 训练与模型超参数
# ========================================

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
LAMBDA_REG = 1.0
LAMBDA_PHYS = 0.01   # 熵正则化权重 (下调以避免约束过强)

EDGE_DIM = 64
NODE_DIM = 128

# 数据增强
AUGMENT_REPEAT = 2
NOISE_STD = 0.05
SCALE_RANGE = (0.8, 1.2)
TIME_SHIFT_MAX = 50
SEED = 42

SAVE_DIR = os.path.join(_THIS_DIR, "outputs")
