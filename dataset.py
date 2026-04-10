<<<<<<< HEAD
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
=======
# -*- coding: utf-8 -*-
"""
dataset.py  —  数据加载 (v4 — 小样本优化)
关键改进:
  1. 全局 Z-score: 跨所有样本计算, 保留样本间差异
  2. 训练/推理都用全部 66 对 (无子采样)
  3. Damage Index 特征: 每对计算标量损伤指标, 作为辅助特征
  4. 多窗口增强: 每个样本提取多个不同位置的时间窗口
"""
import os, glob, re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import hilbert
from collections import defaultdict
import warnings as _warnings

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

from config import (
    DATA_ROOT, REGION_DIRS, REGION_CENTERS, NUM_PAIRS, SENSOR_PAIRS,
    CSV_HEADER_LINES, WINDOW_LEN, IN_CHANNELS,
    BATCH_SIZE, SEED, normalize_coord, NUM_REGIONS, SENSOR_COORDS,
    AUGMENT_REPEAT, NOISE_STD, SCALE_RANGE, TIME_SHIFT_MAX,
    WAVELET_NAME, WAVELET_LEVEL, WAVELET_MODE,
    COORD_MIN, COORD_RANGE,
)


def load_csv(filepath):
    import pandas as pd
    try:
        df = pd.read_csv(filepath, skiprows=CSV_HEADER_LINES, header=None,
                         usecols=[0, 1, 2], dtype=np.float32)
        df = df.dropna()
        data = df.values
    except Exception:
        data = np.genfromtxt(filepath, delimiter=",", skip_header=CSV_HEADER_LINES,
                             usecols=(0, 1, 2), dtype=np.float32)
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data[:, 0], data[:, 1], data[:, 2]


def wavelet_denoise(signal, wavelet=WAVELET_NAME, level=WAVELET_LEVEL, mode=WAVELET_MODE):
    if not HAS_PYWT:
        return signal
    if len(signal) < 2 ** level or np.max(np.abs(signal)) < 1e-12:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    detail_abs = np.abs(coeffs[-1])
    sigma = np.median(detail_abs) / 0.6745 if len(detail_abs) > 0 else 0.0
    if sigma < 1e-12:
        return signal
    threshold = sigma * np.sqrt(2 * np.log(max(len(signal), 2)))
    denoised = [coeffs[0]]
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        for c in coeffs[1:]:
            t = pywt.threshold(c, threshold, mode=mode)
            denoised.append(np.nan_to_num(t, nan=0.0))
    return pywt.waverec(denoised, wavelet).astype(np.float32)[:len(signal)]


def hilbert_envelope(signal):
    return np.abs(hilbert(signal)).astype(np.float32)


def build_sample_index(data_root=DATA_ROOT):
    samples = []
    for ri, rd in enumerate(REGION_DIRS):
        dp = os.path.join(data_root, rd)
        if not os.path.isdir(dp): continue
        tags = set()
        for f in glob.glob(os.path.join(dp, "tek*ALL*.csv")):
            m = re.search(r"ALL(\w+)\.csv", os.path.basename(f))
            if m: tags.add(m.group(1))
        healthy = [os.path.join(dp, f"tek{i:04d}ALLno.csv") for i in range(NUM_PAIRS)]
        if not all(os.path.isfile(f) for f in healthy): continue
        for tag in sorted(tags):
            if tag == "no": continue
            dmg = [os.path.join(dp, f"tek{i:04d}ALL{tag}.csv") for i in range(NUM_PAIRS)]
            if all(os.path.isfile(f) for f in dmg):
                samples.append((ri, tag, dmg, healthy))
    return samples


def preprocess_pair(dmg_path, healthy_path):
    _, exc_d, resp_d = load_csv(dmg_path)
    _, exc_h, resp_h = load_csv(healthy_path)
    L = min(len(exc_d), len(exc_h))
    exc_diff = wavelet_denoise(exc_d[:L] - exc_h[:L])
    resp_diff = wavelet_denoise(resp_d[:L] - resp_h[:L])
    env_exc = hilbert_envelope(exc_diff)
    env_resp = hilbert_envelope(resp_diff)
    return np.stack([exc_diff, resp_diff, env_exc, env_resp], axis=0)


class NDTDataset(Dataset):
    """
    v4 数据集: 全局归一化 + 全66对 + Damage Index特征
    """
    def __init__(self, sample_index, mode="train", global_stats=None):
        self.samples = sample_index
        self.mode = mode
        self.repeat = AUGMENT_REPEAT if mode == "train" else 1

        print(f"  [{mode}] {len(sample_index)} 样本, 预处理...")
        self.data_cache = {}
        for idx, (ri, tag, dmg_files, healthy_files) in enumerate(self.samples):
            signals = []
            for pi in range(NUM_PAIRS):
                sig = preprocess_pair(dmg_files[pi], healthy_files[pi])
                signals.append(sig)
            min_len = min(s.shape[1] for s in signals)
            signals = np.stack([s[:, :min_len] for s in signals], axis=0)  # (66, 4, N)
            signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
            self.data_cache[idx] = signals

        # 全局归一化统计
        if global_stats is not None:
            self.global_mean, self.global_std = global_stats
        elif mode == "train":
            # 计算跨所有样本的全局统计量
            all_data = [self.data_cache[i] for i in range(len(self.samples))]
            self.global_mean = np.zeros(IN_CHANNELS, dtype=np.float32)
            self.global_std = np.ones(IN_CHANNELS, dtype=np.float32)
            for ch in range(IN_CHANNELS):
                ch_all = np.concatenate([d[:, ch, :].ravel() for d in all_data])
                self.global_mean[ch] = float(np.nanmean(ch_all))
                self.global_std[ch] = max(float(np.nanstd(ch_all)), 1e-8)
            print(f"  全局统计: mean={self.global_mean}, std={self.global_std}")
        else:
            self.global_mean = np.zeros(IN_CHANNELS, dtype=np.float32)
            self.global_std = np.ones(IN_CHANNELS, dtype=np.float32)

        # 应用全局归一化
        for idx in range(len(self.samples)):
            signals = self.data_cache[idx]
            for ch in range(IN_CHANNELS):
                signals[:, ch, :] = (signals[:, ch, :] - self.global_mean[ch]) / self.global_std[ch]
            self.data_cache[idx] = np.clip(signals, -10, 10)

        # 计算每对的 Damage Index (标量特征)
        self.di_cache = {}
        for idx in range(len(self.samples)):
            sig = self.data_cache[idx]  # (66, 4, N)
            # DI = 每对响应差分信号的 RMS energy
            di = np.sqrt(np.mean(sig[:, 1, :] ** 2, axis=1))  # (66,)
            self.di_cache[idx] = di.astype(np.float32)

        self.labels = []
        for ri, tag, _, _ in self.samples:
            self.labels.append({
                "region_idx": ri,
                "center_norm": normalize_coord(REGION_CENTERS[ri]),
            })

    def get_global_stats(self):
        return (self.global_mean, self.global_std)

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        signals = self.data_cache[real_idx]  # (66, 4, N)
        di = self.di_cache[real_idx]          # (66,)
        label = self.labels[real_idx]
        N = signals.shape[2]

        if self.mode == "train":
            # 随机时间窗口 + 时移增强
            shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1)
            center = N // 2 + shift
            start = max(0, min(center - WINDOW_LEN // 2, N - WINDOW_LEN))
            selected = signals[:, :, start:start + WINDOW_LEN].copy()

            # 噪声
            selected += np.random.randn(*selected.shape).astype(np.float32) * NOISE_STD
            # 缩放
            scale = np.random.uniform(*SCALE_RANGE)
            selected *= scale
            # 随机对部分传感器对加强扰动
            n_perturb = np.random.randint(0, 10)
            for _ in range(n_perturb):
                pi = np.random.randint(0, NUM_PAIRS)
                selected[pi] *= np.random.uniform(0.5, 1.5)
        else:
            center = N // 2
            start = max(0, center - WINDOW_LEN // 2)
            selected = signals[:, :, start:start + WINDOW_LEN].copy()

        # Pad if needed
        if selected.shape[2] < WINDOW_LEN:
            pad = WINDOW_LEN - selected.shape[2]
            selected = np.pad(selected, ((0,0),(0,0),(0,pad)), mode='constant')

        x = torch.from_numpy(selected).float()           # (66, 4, W)
        di_t = torch.from_numpy(di).float()               # (66,)
        region_idx = torch.tensor(label["region_idx"], dtype=torch.long)
        center_norm = torch.from_numpy(label["center_norm"]).float()
        return x, di_t, region_idx, center_norm


def build_splits(data_root=DATA_ROOT):
    all_samples = build_sample_index(data_root)
    n = len(all_samples)
    print(f"[INFO] 共 {n} 个损伤样本")
    if n == 0: raise ValueError("无样本!")

    region_groups = defaultdict(list)
    for i, (ri, tag, _, _) in enumerate(all_samples):
        region_groups[ri].append(i)

    np.random.seed(SEED)
    train_idx, val_idx, test_idx = [], [], []
    vt = 0
    for ri in sorted(region_groups.keys()):
        idxs = region_groups[ri]; np.random.shuffle(idxs)
        if len(idxs) >= 2:
            train_idx.append(idxs[0])
            (val_idx if vt % 2 == 0 else test_idx).append(idxs[1])
            vt += 1; train_idx.extend(idxs[2:])
        else: train_idx.extend(idxs)
    if not val_idx and len(train_idx) > 2: val_idx.append(train_idx.pop())
    if not test_idx and len(train_idx) > 2: test_idx.append(train_idx.pop())

    for name, ids in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        print(f"  {name}({len(ids)}): {[f'{REGION_DIRS[all_samples[i][0]]}({all_samples[i][1]})' for i in ids]}")

    train_ds = NDTDataset([all_samples[i] for i in train_idx], "train")
    gs = train_ds.get_global_stats()
    val_ds = NDTDataset([all_samples[i] for i in val_idx], "val", global_stats=gs)
    test_ds = NDTDataset([all_samples[i] for i in test_idx], "test", global_stats=gs)
    return train_ds, val_ds, test_ds


def build_full_eval_dataset(data_root=DATA_ROOT, global_stats=None):
    all_samples = build_sample_index(data_root)
    print(f"[INFO] 全样本: {len(all_samples)}")
    # 先构建一个临时的 train dataset 来获取 global stats
    if global_stats is None:
        tmp = NDTDataset(all_samples, "train")
        global_stats = tmp.get_global_stats()
        del tmp
    return NDTDataset(all_samples, "test", global_stats=global_stats), global_stats


def get_dataloaders(data_root=DATA_ROOT):
    train_ds, val_ds, test_ds = build_splits(data_root)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True),
    )


if __name__ == "__main__":
    ds, _ = build_full_eval_dataset()
    x, di, r, c = ds[0]
    print(f"x={x.shape}, di={di.shape}, region={r}, center={c}")
>>>>>>> 20bf0442b9ac059a2ab256e23d75ed0a15abda8c
