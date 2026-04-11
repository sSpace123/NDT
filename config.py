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

# 区域子目录名 (仅用于数据扫描, 不参与分类)
REGION_DIRS = ["左上", "上", "右上", "左中", "中", "右中", "左下", "下", "右下"]

# 各区域损伤中心坐标 (mm), 仅用于生成连续回归标签
DAMAGE_CENTERS = np.array([
    [-83.3,  83.3], [  0.0,  83.3], [ 83.3,  83.3],
    [-83.3,   0.0], [  0.0,   0.0], [ 83.3,   0.0],
    [-83.3, -83.3], [  0.0, -83.3], [ 83.3, -83.3],
], dtype=np.float32)

# 结构坐标极值 (mm)
COORD_MIN = -137.5
COORD_MAX = 137.5
COORD_RANGE = COORD_MAX - COORD_MIN  # 275.0

# 几何 PINN 物理参数
DEAD_ZONE_R = 10.0   # 带状容忍度半径 (mm): 预测点在路径 ±R mm 范围内不受惩罚
                      # 物理含义: 导波散射区域的有效宽度 (而非理想化的一维射线)


def normalize_coord(coord):
    """物理坐标 [-137.5, 137.5] → 归一化 [-1, 1]"""
    return (np.array(coord, dtype=np.float32) - COORD_MIN) / COORD_RANGE * 2.0 - 1.0


def denormalize_coord(norm_coord):
    """归一化 [-1, 1] → 物理坐标 [-137.5, 137.5]"""
    return (np.array(norm_coord, dtype=np.float32) + 1.0) / 2.0 * COORD_RANGE + COORD_MIN


# ========================================
# 2. 数据处理配置 — 动态绝对路径
# ========================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(_THIS_DIR)

WINDOW_HALF_SIZE = 1024
NUM_PAIRS = 66
CSV_HEADER_LINES = 21
IN_CHANNELS = 4

# 小波去噪参数
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 4

# ========================================
# 3. 网络架构参数
# ========================================

EDGE_DIM = 16             # 边特征维度 (极简: 12 样本)
NODE_DIM = 32             # 节点特征维度
TIME_REDUCED_LEN = 128    # 1D 时间轴降维目标长度 (2048 / 16 = 128)
                          # 控制 AvgPool1d 的 kernel_size = 2048 / TIME_REDUCED_LEN

# ========================================
# 4. 训练与 Loss 参数
# ========================================

BATCH_SIZE = 4
EPOCHS = 55
LEARNING_RATE = 5e-4
LAMBDA_REG = 1.0          # MSE 坐标回归损失权重
LAMBDA_PHYS = 0.5         # 几何 PINN 物理正则化权重

# 数据增强 (小样本, 物理约束)
AUGMENT_REPEAT = 20
NOISE_STD = 0.08
SCALE_RANGE = (0.7, 1.3)
TIME_SHIFT_MAX = 5
SEED = 42

SAVE_DIR = os.path.join(_THIS_DIR, "outputs")
