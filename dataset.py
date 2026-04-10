import pandas as pd
import numpy as np
import pywt
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, hilbert
import warnings
from config import FC, FS, WINDOW_HALF_SIZE

# 屏蔽 fastdtw 可能产生的无关警告
warnings.filterwarnings('ignore')

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("Warning: Please install fastdtw library using: pip install fastdtw")

# ========================================
# 二、 针对复杂结构的预处理 & 三、 高级数据增强
# ========================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """四阶 Butterworth 带通滤波器: 限制能量于主模态 [0.5fc, 1.5fc]"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def wavelet_denoise(data, wavelet='db4', level=4):
    """
    采用 db4 小波进行多层分解。
    计算细节系数的 MAD(绝对中位差)来进行自适应阈值软去噪。
    """
    coeffs = pywt.wavedec(data, wavelet, mode='per', level=level)
    # 取最后一层细节系数计算 MAD 并生成噪声阈值
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    # Soft Thresholding
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode='per')

def extract_envelope(data):
    """提取时域信号的绝对包络: |hilbert(s)|"""
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def get_cwt_spectrogram(data, fs=FS, base_freq=FC):
    """
    CWT 时频图 (Morlet小波):
    在此，我们生成小范围的一组物理敏感频率对应的尺度
    返回二维频域时频图 [F, T]
    """
    # 以 fs, base_freq 为基准生成频率尺度，这里默认生成 F=32 个频带
    # 物理上有意义的区间大致在 [0.5 * FC, 1.5 * FC]
    freqs = np.linspace(0.5 * base_freq, 1.5 * base_freq, 32)
    # 尺度公式: scale = wavelet_center_frequency * fs / target_frequency
    wavelet = 'cgau1' # Complex Gaussian/Morlet 近似
    # Pywavelets 核心中心频率 (cgau1 约 0.5)
    central_freq = pywt.central_frequency(wavelet) 
    scales = central_freq * fs / freqs
    
    # [F, T]
    cwtmatr, _ = pywt.cwt(data, scales, wavelet)
    return np.abs(cwtmatr) # 取振幅图

def preprocess_pair(base_csv_path, dmg_csv_path):
    """
    单个路径上的两条信号处理 Pipeline。
    返回: [4个特征通道的 Numpy CWT 张量], shape => [4, F, 2048]
    - 通道0: diff_exc (计算激励差别)
    - 通道1: diff_resp (计算响应差别)
    - 通道2: env_exc (激励包络)
    - 通道3: env_resp (响应包络)
    """
    # 1. 读入原始信号，跳过前21行 (示波器头文件)
    base_df = pd.read_csv(base_csv_path, skiprows=21, usecols=[1, 2], names=["excitation", "response"])
    dmg_df = pd.read_csv(dmg_csv_path, skiprows=21, usecols=[1, 2], names=["excitation", "response"])
    
    base_exc = base_df["excitation"].values.astype(np.float32)
    base_resp = base_df["response"].values.astype(np.float32)
    dmg_exc = dmg_df["excitation"].values.astype(np.float32)
    dmg_resp = dmg_df["response"].values.astype(np.float32)
    
    # 2. 鲁棒对齐 (DTW) - 由于响应信号可能因扭转和温度变化发生畸变
    # 注意: fastdtw 会消耗较多计算，实际工业场景可限制搜寻窗或降采样处理
    # 限于计算性能，降采样后搜索对齐关系，再映射回原信号
    # 但为严谨本例使用完整的一维欧氏对齐。这里将 dmg_resp 对齐到 base_resp
    distance, path = fastdtw(base_resp, dmg_resp, dist=euclidean)
    aligned_dmg_resp = np.zeros_like(base_resp)
    # 映射 (对于多对1映射取均值，这里简略直接覆盖索引)
    for p_base, p_dmg in path:
        if p_base < len(aligned_dmg_resp):
            aligned_dmg_resp[p_base] = dmg_resp[p_dmg]
            
    # 同理对齐激发信号 (尽管变动极小)
    aligned_dmg_exc = dmg_exc[:len(base_exc)] # 激发信号强刚性，通常无需 DTW
    
    # 3. 差分计算
    diff_exc = aligned_dmg_exc - base_exc
    diff_resp = aligned_dmg_resp - base_resp
    
    # 4. 物理窄带滤波 [0.5fc, 1.5fc]
    diff_exc = butter_bandpass_filter(diff_exc, 0.5 * FC, 1.5 * FC, FS, order=4)
    diff_resp = butter_bandpass_filter(diff_resp, 0.5 * FC, 1.5 * FC, FS, order=4)
    
    # 5. 去噪 MAD Soft threshold
    diff_exc = wavelet_denoise(diff_exc)
    diff_resp = wavelet_denoise(diff_resp)
    
    # 6. 包络提取
    env_exc = extract_envelope(diff_exc)
    env_resp = extract_envelope(diff_resp)
    
    # 7. 物理窗口对齐裁剪: 基于包络最大峰值向两侧固定截取 2048 点
    peak_idx = np.argmax(env_resp)
    length = len(env_resp)
    start_idx = max(0, peak_idx - WINDOW_HALF_SIZE)
    end_idx = min(length, peak_idx + WINDOW_HALF_SIZE)
    
    # 如果处于首尾段导致长度不足 2048，进行补零
    pad_left = max(0, WINDOW_HALF_SIZE - peak_idx)
    pad_right = max(0, (peak_idx + WINDOW_HALF_SIZE) - length)
    
    diff_exc_cropped = np.pad(diff_exc[start_idx:end_idx], (pad_left, pad_right), 'constant')
    diff_resp_cropped = np.pad(diff_resp[start_idx:end_idx], (pad_left, pad_right), 'constant')
    env_exc_cropped = np.pad(env_exc[start_idx:end_idx], (pad_left, pad_right), 'constant')
    env_resp_cropped = np.pad(env_resp[start_idx:end_idx], (pad_left, pad_right), 'constant')
    
    # 8. 时频变换 (CWT) -> [4, F=32, T=2048]
    # 对特征信号全部做 CWT (或者仅对直接信号)。依要求构造 4通道。
    feat0 = get_cwt_spectrogram(diff_exc_cropped)
    feat1 = get_cwt_spectrogram(diff_resp_cropped)
    feat2 = get_cwt_spectrogram(env_exc_cropped)
    feat3 = get_cwt_spectrogram(env_resp_cropped)
    
    tensor_cwt = np.stack([feat0, feat1, feat2, feat3], axis=0) # [4, 32, 2048]
    
    # Z-score 全局标准化与 Clip [-10, 10]
    mu = np.mean(tensor_cwt)
    sigma = np.std(tensor_cwt) + 1e-8
    tensor_cwt = (tensor_cwt - mu) / sigma
    tensor_cwt = np.clip(tensor_cwt, -10.0, 10.0)
    
    return tensor_cwt

class SHMGraphDataset(Dataset):
    def __init__(self, sample_list, augment=True):
        """
        :param sample_list: [(paths_dict, target_coord, target_cls), ...]
               paths_dict: key为边索引(0~65), value为(base_csv, dmg_csv)
        """
        self.sample_list = sample_list
        self.augment = augment

    def __len__(self):
        return len(self.sample_list)

    @staticmethod
    def _baseline_stretch(tensor_4d, stretch_ratio):
        """
        温度效应模拟 (Baseline Stretch):
        通过线性插值对时间轴进行微小拉伸/压缩，模拟环境温度对波速的影响。
        :param tensor_4d: [4, F, T] 单条边的特征
        :param stretch_ratio: 拉伸比例，例如 1.003 表示 0.3% 拉伸
        :return: 拉伸后的 [4, F, T] (与原始长度相同，通过裁剪/补零保持)
        """
        C, F_, T = tensor_4d.shape
        new_T = int(T * stretch_ratio)
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, new_T)
        result = np.zeros_like(tensor_4d)
        for c in range(C):
            for f in range(F_):
                stretched = np.interp(x_new, x_old, tensor_4d[c, f, :])
                # 裁剪或补零使长度保持 T
                if new_T >= T:
                    result[c, f, :] = stretched[:T]
                else:
                    result[c, f, :new_T] = stretched
        return result

    @staticmethod
    def mixup(data1, data2, coord1, coord2, alpha=0.3):
        """
        Mixup (波形混合): 将同一区域内的损伤特征进行线性加权混合。
        :param data1: [66, 4, F, T] 样本1
        :param data2: [66, 4, F, T] 样本2 (应属于同区域)
        :param coord1: [2] 样本1坐标
        :param coord2: [2] 样本2坐标
        :param alpha: Beta分布参数
        :return: 混合后的 (data_mixed, coord_mixed)
        """
        lam = np.random.beta(alpha, alpha)
        data_mixed = lam * data1 + (1 - lam) * data2
        coord_mixed = lam * coord1 + (1 - lam) * coord2
        return data_mixed.astype(np.float32), coord_mixed.astype(np.float32)

    def _apply_augs(self, sample_tensor):
        """
        [ 三、 高级数据增强 ] (各项独立以 p=0.5 触发)
        sample_tensor: [66, 4, 32, 2048] Numpy
        """
        E, C, F_, T = sample_tensor.shape  # 66, 4, 32, 2048

        # 1. 温度效应模拟 (Baseline Stretch): 随机 0.1%~0.5% 拉伸或压缩
        if np.random.rand() < 0.5:
            stretch_pct = np.random.uniform(0.001, 0.005)
            stretch_ratio = 1.0 + np.random.choice([-1, 1]) * stretch_pct
            for e in range(E):
                if np.any(sample_tensor[e]):  # 跳过全零(已 dropout)的边
                    sample_tensor[e] = self._baseline_stretch(sample_tensor[e], stretch_ratio)

        # 2. 传感器 Dropout: 随机 Mask 掉 1~3 条路径 (对应边设为0)
        if np.random.rand() < 0.5:
            num_drop = np.random.randint(1, 4)
            drop_edges = np.random.choice(E, num_drop, replace=False)
            sample_tensor[drop_edges, :, :, :] = 0.0

        # 3. 物理微扰: ±50点时移补零, 幅度缩放 U(0.8, 1.2), 添加微小高斯噪声
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 51)
            if shift > 0:  # 信号右移
                sample_tensor[:, :, :, shift:] = sample_tensor[:, :, :, :-shift].copy()
                sample_tensor[:, :, :, :shift] = 0
            elif shift < 0:  # 左移
                shift_abs = abs(shift)
                sample_tensor[:, :, :, :-shift_abs] = sample_tensor[:, :, :, shift_abs:].copy()
                sample_tensor[:, :, :, -shift_abs:] = 0

            # 幅度缩放
            scale = np.random.uniform(0.8, 1.2)
            sample_tensor *= scale

            # 高斯白噪声 (sigma=0.05)
            noise = np.random.normal(0, 0.05, sample_tensor.shape).astype(np.float32)
            sample_tensor += noise

        return sample_tensor

    def __getitem__(self, idx):
        paths_dict, target_coord, target_cls = self.sample_list[idx]
        
        # 构建 66 条边的输入张量 [66, 4, 32, 2048]
        graph_data = np.zeros((66, 4, 32, 2048), dtype=np.float32)
        
        for edge_idx in range(66):
            if edge_idx in paths_dict:
                b_path, d_path = paths_dict[edge_idx]
                # 执行所有的数字图像/信号获取预处理
                # (注意：工业使用中，建议预先跑完预处理并存为 .pt / .npy 读取，在线处理会导致 Dataloader 极品慢)
                graph_data[edge_idx] = preprocess_pair(b_path, d_path)
                
        # 执行概率0.5的数据增强 (除 Mixup 以外)
        # Mixup 通常在训练的 batch 层面进行 (model.forward 之前融合两个batch的数据) 
        if self.augment:
            graph_data = self._apply_augs(graph_data)
            
        t_data = torch.from_numpy(graph_data)
        t_coord = torch.tensor(target_coord, dtype=torch.float32)
        t_cls = torch.tensor(target_cls, dtype=torch.long)
        
        return t_data, t_coord, t_cls

if __name__ == '__main__':
    # 模拟构建假数据路径以测试接口是否连通
    print("Dataset pipeline is ready.")
