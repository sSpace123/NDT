import numpy as np

# ========================================
# 1. 物理参数与传感器配置
# ========================================

# 采样率 (Unit: Hz)
FS = 1e7  # 10 MHz

# 激励信号中心频率 (Unit: Hz)
FC = 100e3  # 100 kHz 五周期 Tone burst

# 初始导波群速度参考基准 (Unit: m/s)
CG = 3000.0  

# 传感器左、中、右布置三列，每列四个传感器 (12个)
# 左侧: 0~5; 右侧: 6~11
# 注意：这只是为了说明二分图的两边。这里把左上方的都算作左侧。
# 坐标单位: mm
SENSOR_COORDS = np.array([
    [-125.0,  125.0], [-125.0,   75.0], [-125.0,   25.0],
    [-125.0,  -25.0], [-125.0,  -75.0], [-125.0, -125.0],
    [ 125.0,  125.0], [ 125.0,   75.0], [ 125.0,   25.0],
    [ 125.0,  -25.0], [ 125.0,  -75.0], [ 125.0, -125.0],
], dtype=np.float32)

# 网格/区域划分名称
REGION_NAMES = ["左上", "上", "右上", "左中", "中", "右中", "左下", "下", "右下"]
NUM_REGIONS = 9

# 区域中心地标 (用于展示和粗分类) mm
REGION_CENTERS = np.array([
    [-83.3,  83.3], [  0.0,  83.3], [ 83.3,  83.3],
    [-83.3,   0.0], [  0.0,   0.0], [ 83.3,   0.0],
    [-83.3, -83.3], [  0.0, -83.3], [ 83.3, -83.3],
], dtype=np.float32)

# 结构的坐标极值定义 (Unit: mm)
COORD_MIN = -137.5
COORD_MAX = 137.5
COORD_RANGE = COORD_MAX - COORD_MIN

def normalize_coord(coord):
    """ 将物理坐标从 [-137.5, 137.5] 归一化到 [-1, 1] """
    return (np.array(coord, dtype=np.float32) - COORD_MIN) / (COORD_RANGE) * 2.0 - 1.0

def denormalize_coord(norm_coord):
    """ 将网络输出坐标从 [-1, 1] 反归一化到真实物理范围 """
    return (np.array(norm_coord, dtype=np.float32) + 1.0) / 2.0 * COORD_RANGE + COORD_MIN

# ========================================
# 2. 数据处理与存储配置
# ========================================

# 数据根目录及区域子目录
DATA_ROOT = "../data" 
REGION_DIRS = ["region_0", "region_1", "region_2", "region_3", "region_4", "region_5", "region_6", "region_7", "region_8"]

# 信号截取窗口长度 (中心峰值两侧各截取，总长 2048)
WINDOW_HALF_SIZE = 1024
WINDOW_SIZE = WINDOW_HALF_SIZE * 2
WINDOW_LEN = 2048

# 信号与 CSV 参数
NUM_PAIRS = 66
CSV_HEADER_LINES = 21

# 特征提取
IN_CHANNELS = 4
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 4
WAVELET_MODE = 'per'

# ========================================
# 3. 训练与模型超参数 (统一配置)
# ========================================

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
LAMBDA_REG = 1.0
LAMBDA_PHYS = 0.01  # 下调物理约束权重
NUM_CLASSES = 9

# 网络尺寸配置
EDGE_DIM = 64
NODE_DIM = 128
NUM_NODES = 12

# 数据增强
AUGMENT_REPEAT = 2
NOISE_STD = 0.05
SCALE_RANGE = (0.8, 1.2)
TIME_SHIFT_MAX = 50
SEED = 42

SAVE_DIR = "outputs"
PLOT_DIR = "outputs"
EPSILON = 1e-8
