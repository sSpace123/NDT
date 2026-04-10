import os
import glob
import re
import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, hilbert
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("Warning: Please install fastdtw library using: pip install fastdtw")

from config import (
    DATA_ROOT, REGION_DIRS, REGION_CENTERS, NUM_PAIRS, CSV_HEADER_LINES, 
    WINDOW_HALF_SIZE, IN_CHANNELS, BATCH_SIZE, SEED, normalize_coord, 
    AUGMENT_REPEAT, NOISE_STD, SCALE_RANGE, TIME_SHIFT_MAX,
    FS, FC
)

# ========================================
# 预计算 36 边图映射 (Left: 0~5, Right: 6~11)
# ========================================
BIPARTITE_INDICES = []
_idx = 0
for i in range(12):
    for j in range(i+1, 12):
        if i < 6 and j >= 6:
            BIPARTITE_INDICES.append(_idx)
        _idx += 1

# ========================================
# CWT特征提取与信号预处理
# ========================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def wavelet_denoise(data, wavelet='db4', level=4):
    if np.max(np.abs(data)) < 1e-12:
        return data
    coeffs = pywt.wavedec(data, wavelet, mode='per', level=level)
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
    if sigma < 1e-12:
        return data
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode='per')

def extract_envelope(data):
    analytic_signal = hilbert(data)
    return np.abs(analytic_signal)

def get_cwt_spectrogram(data, fs=FS, base_freq=FC):
    freqs = np.linspace(0.5 * base_freq, 1.5 * base_freq, 32)
    wavelet = 'cgau1' 
    central_freq = pywt.central_frequency(wavelet) 
    scales = central_freq * fs / freqs
    cwtmatr, _ = pywt.cwt(data, scales, wavelet)
    return np.abs(cwtmatr) 

def preprocess_pair(base_csv_path, dmg_csv_path):
    # 读入原始信号，跳过头文件
    try:
        base_df = pd.read_csv(base_csv_path, skiprows=CSV_HEADER_LINES, usecols=[1, 2], names=["excitation", "response"])
        dmg_df = pd.read_csv(dmg_csv_path, skiprows=CSV_HEADER_LINES, usecols=[1, 2], names=["excitation", "response"])
    except Exception:
        base_data = np.genfromtxt(base_csv_path, delimiter=",", skip_header=CSV_HEADER_LINES, usecols=(1, 2))
        dmg_data = np.genfromtxt(dmg_csv_path, delimiter=",", skip_header=CSV_HEADER_LINES, usecols=(1, 2))
        base_df = pd.DataFrame(base_data, columns=["excitation", "response"])
        dmg_df = pd.DataFrame(dmg_data, columns=["excitation", "response"])
        
    base_exc = base_df["excitation"].values.astype(np.float32)
    base_resp = base_df["response"].values.astype(np.float32)
    dmg_exc = dmg_df["excitation"].values.astype(np.float32)
    dmg_resp = dmg_df["response"].values.astype(np.float32)
    
    # 鲁棒对齐 (考虑 fastdtw 开销，简化为直接截断到最短)
    # 此处为简化处理，真实工程中如果强烈需要使用 DTW 可开启，这里使用直接对齐
    min_len = min(len(base_exc), len(dmg_exc), len(base_resp), len(dmg_resp))
    base_exc, base_resp = base_exc[:min_len], base_resp[:min_len]
    dmg_exc, dmg_resp = dmg_exc[:min_len], dmg_resp[:min_len]
    
    aligned_dmg_exc = dmg_exc
    aligned_dmg_resp = dmg_resp
    
    # 差分
    diff_exc = aligned_dmg_exc - base_exc
    diff_resp = aligned_dmg_resp - base_resp
    
    # 物理窄带滤波
    diff_exc = butter_bandpass_filter(diff_exc, 0.5 * FC, 1.5 * FC, FS, order=4)
    diff_resp = butter_bandpass_filter(diff_resp, 0.5 * FC, 1.5 * FC, FS, order=4)
    
    # 去噪
    diff_exc = wavelet_denoise(diff_exc)
    diff_resp = wavelet_denoise(diff_resp)
    
    # 包络
    env_exc = extract_envelope(diff_exc)
    env_resp = extract_envelope(diff_resp)
    
    # 物理窗口对齐裁剪: 基于包络最大峰值向两侧固定截取 2048 点
    peak_idx = np.argmax(env_resp)
    length = len(env_resp)
    start_idx = max(0, peak_idx - WINDOW_HALF_SIZE)
    end_idx = min(length, peak_idx + WINDOW_HALF_SIZE)
    
    pad_left = max(0, WINDOW_HALF_SIZE - peak_idx)
    pad_right = max(0, (peak_idx + WINDOW_HALF_SIZE) - length)
    
    diff_exc_cropped = np.pad(diff_exc[start_idx:end_idx], (pad_left, pad_right), 'constant')
    diff_resp_cropped = np.pad(diff_resp[start_idx:end_idx], (pad_left, pad_right), 'constant')
    env_exc_cropped = np.pad(env_exc[start_idx:end_idx], (pad_left, pad_right), 'constant')
    env_resp_cropped = np.pad(env_resp[start_idx:end_idx], (pad_left, pad_right), 'constant')
    
    # CWT 特征 -> [4, 32, 2048]
    feat0 = get_cwt_spectrogram(diff_exc_cropped)
    feat1 = get_cwt_spectrogram(diff_resp_cropped)
    feat2 = get_cwt_spectrogram(env_exc_cropped)
    feat3 = get_cwt_spectrogram(env_resp_cropped)
    
    tensor_cwt = np.stack([feat0, feat1, feat2, feat3], axis=0).astype(np.float32)
    return tensor_cwt

# ========================================
# v4 版本结构的数据目录扫描与数据集构建
# ========================================

def build_sample_index(data_root=DATA_ROOT):
    samples = []
    print(f"Scanning DATA_ROOT: {data_root}")
    if not os.path.exists(data_root):
        print(f"Directory {data_root} not found. Generating dummy paths for testing.")
        return []
        
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

class NDTDataset(Dataset):
    def __init__(self, sample_index, mode="train", global_stats=None, is_dummy=False):
        self.samples = sample_index
        self.mode = mode
        self.repeat = AUGMENT_REPEAT if mode == "train" else 1
        self.is_dummy = is_dummy
        
        self.data_cache = {}
        if not is_dummy:
            print(f"  [{mode}] {len(sample_index)} 样本, 提取 36 条 Bipartite 边 CWT 特征...")
            for idx, (ri, tag, dmg_files, healthy_files) in enumerate(self.samples):
                signals = []
                for pi in BIPARTITE_INDICES:
                    sig = preprocess_pair(dmg_files[pi], healthy_files[pi]) # [4, 32, 2048]
                    signals.append(sig)
                signals = np.stack(signals, axis=0)  # [36, 4, 32, 2048]
                signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
                self.data_cache[idx] = signals

            # 全局归一化统计 (Z-score)
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            elif mode == "train" and len(self.samples) > 0:
                all_data = np.stack([self.data_cache[i] for i in range(len(self.samples))]) # [N, 36, 4, 32, 2048]
                self.global_mean = np.mean(all_data, axis=(0, 1, 3, 4), keepdims=True) # [1, 1, 4, 1, 1]
                self.global_std = np.std(all_data, axis=(0, 1, 3, 4), keepdims=True) + 1e-8
                print(f"  全局特征统计提取完成")
            else:
                self.global_mean = np.zeros((1, 1, IN_CHANNELS, 1, 1), dtype=np.float32)
                self.global_std = np.ones((1, 1, IN_CHANNELS, 1, 1), dtype=np.float32)

            for idx in range(len(self.samples)):
                signals = self.data_cache[idx]
                signals = (signals - self.global_mean.squeeze(0)) / self.global_std.squeeze(0)
                self.data_cache[idx] = np.clip(signals, -10, 10).astype(np.float32)
        else:
            self.global_mean = np.zeros((1, 1, IN_CHANNELS, 1, 1), dtype=np.float32)
            self.global_std = np.ones((1, 1, IN_CHANNELS, 1, 1), dtype=np.float32)

        self.labels = []
        if not is_dummy:
            for ri, tag, _, _ in self.samples:
                self.labels.append({
                    "region_idx": ri,
                    "center_norm": normalize_coord(REGION_CENTERS[ri]),
                })
        else:
            for i in range(len(self.samples)):
                self.labels.append({
                    "region_idx": np.random.randint(0, 9),
                    "center_norm": normalize_coord(np.random.normal(0, 50, 2)),
                })

    def get_global_stats(self):
        return (self.global_mean, self.global_std)

    def __len__(self):
        return len(self.samples) * self.repeat

    def _baseline_stretch(self, tensor_4d, stretch_ratio):
        C, F_, T = tensor_4d.shape
        new_T = int(T * stretch_ratio)
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, new_T)
        result = np.zeros_like(tensor_4d)
        for c in range(C):
            for f in range(F_):
                stretched = np.interp(x_new, x_old, tensor_4d[c, f, :])
                if new_T >= T:
                    result[c, f, :] = stretched[:T]
                else:
                    result[c, f, :new_T] = stretched
        return result

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        
        if self.is_dummy:
            # 伪数据 [36, 4, 32, 2048]
            selected = np.random.randn(36, IN_CHANNELS, 32, WINDOW_HALF_SIZE*2).astype(np.float32)
        else:
            selected = self.data_cache[real_idx].copy() # [36, 4, 32, 2048]
            
            if self.mode == "train":
                # 增强
                if np.random.rand() < 0.5:
                    stretch_ratio = 1.0 + np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005)
                    for e in range(36):
                        selected[e] = self._baseline_stretch(selected[e], stretch_ratio)
                    
                if np.random.rand() < 0.5:
                    num_drop = np.random.randint(1, 4)
                    drop_edges = np.random.choice(36, num_drop, replace=False)
                    selected[drop_edges, :, :, :] = 0.0
                    
                if np.random.rand() < 0.5:
                    shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX)
                    if shift > 0:
                        selected[:, :, :, shift:] = selected[:, :, :, :-shift].copy()
                        selected[:, :, :, :shift] = 0
                    elif shift < 0:
                        shift_abs = abs(shift)
                        selected[:, :, :, :-shift_abs] = selected[:, :, :, shift_abs:].copy()
                        selected[:, :, :, -shift_abs:] = 0
                        
                    selected *= np.random.uniform(*SCALE_RANGE)
                    selected += np.random.normal(0, NOISE_STD, selected.shape).astype(np.float32)

        label = self.labels[real_idx]
        x = torch.from_numpy(selected).float() # [36, 4, 32, 2048]
        region_idx = torch.tensor(label["region_idx"], dtype=torch.long)
        center_norm = torch.from_numpy(label["center_norm"]).float() # [-1, 1] Normalized Coords
        
        return x, center_norm, region_idx

def build_splits(data_root=DATA_ROOT):
    all_samples = build_sample_index(data_root)
    n = len(all_samples)
    
    if n == 0:
        print("[WARNING] Data not found! Generating dummy splits for testing the architecture.")
        dummy_samples = list(range(32))
        train_ds = NDTDataset(dummy_samples[:24], "train", is_dummy=True)
        val_ds = NDTDataset(dummy_samples[24:], "val", is_dummy=True)
        test_ds = NDTDataset(dummy_samples[24:], "test", is_dummy=True)
        return train_ds, val_ds, test_ds

    region_groups = defaultdict(list)
    for i, (ri, tag, _, _) in enumerate(all_samples):
        region_groups[ri].append(i)

    np.random.seed(SEED)
    train_idx, val_idx, test_idx = [], [], []
    vt = 0
    for ri in sorted(region_groups.keys()):
        idxs = region_groups[ri]
        np.random.shuffle(idxs)
        if len(idxs) >= 2:
            train_idx.append(idxs[0])
            (val_idx if vt % 2 == 0 else test_idx).append(idxs[1])
            vt += 1
            train_idx.extend(idxs[2:])
        else: train_idx.extend(idxs)
        
    train_ds = NDTDataset([all_samples[i] for i in train_idx], "train")
    gs = train_ds.get_global_stats()
    val_ds = NDTDataset([all_samples[i] for i in val_idx], "val", global_stats=gs)
    test_ds = NDTDataset([all_samples[i] for i in test_idx], "test", global_stats=gs)
    return train_ds, val_ds, test_ds

def get_dataloaders(data_root=DATA_ROOT):
    train_ds, val_ds, test_ds = build_splits(data_root)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False),
    )
