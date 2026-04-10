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

from config import (
    DATA_ROOT, REGION_DIRS, REGION_CENTERS, NUM_PAIRS, CSV_HEADER_LINES,
    WINDOW_HALF_SIZE, IN_CHANNELS, BATCH_SIZE, SEED, normalize_coord,
    AUGMENT_REPEAT, NOISE_STD, SCALE_RANGE, TIME_SHIFT_MAX,
    FS, FC, NUM_CLASSES, WAVELET_NAME, WAVELET_LEVEL,
)

# ========================================
# 预计算 36 边二分图索引映射 (Left 0~5 × Right 6~11)
# 从原始 66 对 C(12,2) 中精确提取
# ========================================
BIPARTITE_INDICES = []
_idx = 0
for _i in range(12):
    for _j in range(_i + 1, 12):
        if _i < 6 and _j >= 6:
            BIPARTITE_INDICES.append(_idx)
        _idx += 1
assert len(BIPARTITE_INDICES) == 36

NUM_BIPARTITE_EDGES = 36

# ========================================
# 信号预处理与 CWT 特征提取
# ========================================

def _butter_bandpass(data, lowcut, highcut, fs, order=4):
    """四阶 Butterworth 带通滤波"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


def _wavelet_denoise(data):
    """db4 小波 MAD 自适应软阈值去噪"""
    if np.max(np.abs(data)) < 1e-12:
        return data
    coeffs = pywt.wavedec(data, WAVELET_NAME, mode='per', level=WAVELET_LEVEL)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma < 1e-12:
        return data
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, WAVELET_NAME, mode='per')


def _get_cwt(data):
    """CWT 时频图 (cgau1 小波), 返回 [32, T]"""
    freqs = np.linspace(0.5 * FC, 1.5 * FC, 32)
    wavelet = 'cgau1'
    scales = pywt.central_frequency(wavelet) * FS / freqs
    cwtmatr, _ = pywt.cwt(data, scales, wavelet)
    return np.abs(cwtmatr)


def preprocess_pair(healthy_csv, damage_csv):
    """
    单条路径的完整 Pipeline: 读取 → 差分 → 滤波 → 去噪 → 包络 → 窗口裁剪 → CWT
    返回: [4, 32, 2048]
    """
    # 读取 CSV
    try:
        base_df = pd.read_csv(healthy_csv, skiprows=CSV_HEADER_LINES,
                              usecols=[1, 2], names=["excitation", "response"])
        dmg_df = pd.read_csv(damage_csv, skiprows=CSV_HEADER_LINES,
                             usecols=[1, 2], names=["excitation", "response"])
    except Exception:
        base_data = np.genfromtxt(healthy_csv, delimiter=",",
                                  skip_header=CSV_HEADER_LINES, usecols=(1, 2))
        dmg_data = np.genfromtxt(damage_csv, delimiter=",",
                                  skip_header=CSV_HEADER_LINES, usecols=(1, 2))
        base_df = pd.DataFrame(base_data, columns=["excitation", "response"])
        dmg_df = pd.DataFrame(dmg_data, columns=["excitation", "response"])

    # 截断到共同长度 & 清洗异常值 (示波器溢出会产生 inf)
    min_len = min(len(base_df), len(dmg_df))
    base_exc = np.nan_to_num(base_df["excitation"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    base_resp = np.nan_to_num(base_df["response"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmg_exc = np.nan_to_num(dmg_df["excitation"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmg_resp = np.nan_to_num(dmg_df["response"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 差分
    diff_exc = dmg_exc - base_exc
    diff_resp = dmg_resp - base_resp

    # 带通滤波 [0.5fc, 1.5fc]
    diff_exc = _butter_bandpass(diff_exc, 0.5 * FC, 1.5 * FC, FS)
    diff_resp = _butter_bandpass(diff_resp, 0.5 * FC, 1.5 * FC, FS)

    # 小波去噪
    diff_exc = _wavelet_denoise(diff_exc)
    diff_resp = _wavelet_denoise(diff_resp)

    # Hilbert 包络
    env_exc = np.abs(hilbert(diff_exc))
    env_resp = np.abs(hilbert(diff_resp))

    # 峰值对齐窗口裁剪 (±1024 = 2048 点)
    peak = np.argmax(env_resp)
    length = len(env_resp)
    start = max(0, peak - WINDOW_HALF_SIZE)
    end = min(length, peak + WINDOW_HALF_SIZE)
    pad_l = max(0, WINDOW_HALF_SIZE - peak)
    pad_r = max(0, (peak + WINDOW_HALF_SIZE) - length)

    def _crop_pad(sig):
        return np.pad(sig[start:end], (pad_l, pad_r), 'constant')

    # CWT → [4, 32, 2048]
    tensor_cwt = np.stack([
        _get_cwt(_crop_pad(diff_exc)),
        _get_cwt(_crop_pad(diff_resp)),
        _get_cwt(_crop_pad(env_exc)),
        _get_cwt(_crop_pad(env_resp)),
    ], axis=0).astype(np.float32)

    return tensor_cwt


# ========================================
# 数据目录扫描
# ========================================

def build_sample_index(data_root=DATA_ROOT):
    """扫描 data_root 下各区域子目录, 返回 [(region_idx, tag, dmg_files, healthy_files), ...]"""
    samples = []
    if not os.path.exists(data_root):
        print(f"[WARN] DATA_ROOT '{data_root}' not found.")
        return samples

    for ri, rd in enumerate(REGION_DIRS):
        dp = os.path.join(data_root, rd)
        if not os.path.isdir(dp):
            continue
        # 收集所有 tag
        tags = set()
        for f in glob.glob(os.path.join(dp, "tek*ALL*.csv")):
            m = re.search(r"ALL(\w+)\.csv", os.path.basename(f))
            if m:
                tags.add(m.group(1))
        # 健康基线文件
        healthy = [os.path.join(dp, f"tek{i:04d}ALLno.csv") for i in range(NUM_PAIRS)]
        if not all(os.path.isfile(f) for f in healthy):
            continue
        for tag in sorted(tags):
            if tag == "no":
                continue
            dmg = [os.path.join(dp, f"tek{i:04d}ALL{tag}.csv") for i in range(NUM_PAIRS)]
            if all(os.path.isfile(f) for f in dmg):
                samples.append((ri, tag, dmg, healthy))
    return samples


# ========================================
# Dataset
# ========================================

class NDTDataset(Dataset):
    """
    加载 36 条二分图边的 CWT 特征, 返回 (x, coord_norm, region_idx).
    - x: [36, 4, 32, 2048]
    - coord_norm: [2]  归一化到 [-1, 1]
    - region_idx: scalar  0~8
    """

    def __init__(self, sample_index, mode="train", global_stats=None, is_dummy=False):
        self.samples = sample_index
        self.mode = mode
        self.repeat = AUGMENT_REPEAT if mode == "train" else 1
        self.is_dummy = is_dummy

        # --- 预处理并缓存 ---
        self.data_cache = {}
        if not is_dummy:
            print(f"  [{mode}] {len(sample_index)} 样本, 提取 {NUM_BIPARTITE_EDGES} 条边 CWT 特征...")
            for idx, (ri, tag, dmg_files, healthy_files) in enumerate(self.samples):
                signals = np.stack([
                    preprocess_pair(healthy_files[pi], dmg_files[pi])
                    for pi in BIPARTITE_INDICES
                ], axis=0)  # [36, 4, 32, 2048]
                self.data_cache[idx] = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 全局 Z-score 归一化 ---
        if global_stats is not None:
            self.global_mean, self.global_std = global_stats
        elif not is_dummy and mode == "train" and len(self.samples) > 0:
            all_data = np.stack(list(self.data_cache.values()))  # [N, 36, 4, 32, 2048]
            self.global_mean = np.mean(all_data, axis=(0, 1, 3, 4), keepdims=False)  # [4]
            self.global_std = np.std(all_data, axis=(0, 1, 3, 4), keepdims=False) + 1e-8  # [4]
            print(f"  全局统计: mean={self.global_mean}, std={self.global_std}")
        else:
            self.global_mean = np.zeros(IN_CHANNELS, dtype=np.float32)
            self.global_std = np.ones(IN_CHANNELS, dtype=np.float32)

        # 应用归一化
        if not is_dummy:
            mean = self.global_mean[np.newaxis, :, np.newaxis, np.newaxis]  # [1, 4, 1, 1]
            std = self.global_std[np.newaxis, :, np.newaxis, np.newaxis]
            for idx in range(len(self.samples)):
                normed = (self.data_cache[idx] - mean) / std
                self.data_cache[idx] = np.clip(normed, -10, 10).astype(np.float32)

        # --- 标签 ---
        self.labels = []
        if not is_dummy:
            for ri, tag, _, _ in self.samples:
                self.labels.append({
                    "region_idx": ri,
                    "center_norm": normalize_coord(REGION_CENTERS[ri]),
                })
        else:
            rng = np.random.RandomState(SEED)
            for _ in range(len(self.samples)):
                self.labels.append({
                    "region_idx": rng.randint(0, NUM_CLASSES),
                    "center_norm": normalize_coord(rng.normal(0, 50, 2)),
                })

    def get_global_stats(self):
        return (self.global_mean, self.global_std)

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)

        if self.is_dummy:
            selected = np.random.randn(NUM_BIPARTITE_EDGES, IN_CHANNELS, 32,
                                       WINDOW_HALF_SIZE * 2).astype(np.float32) * 0.1
        else:
            selected = self.data_cache[real_idx].copy()

            if self.mode == "train":
                # 传感器 Dropout: mask 1~3 条边
                if np.random.rand() < 0.5:
                    drop = np.random.choice(NUM_BIPARTITE_EDGES,
                                            np.random.randint(1, 4), replace=False)
                    selected[drop] = 0.0

                # 物理微扰: 时移 + 幅度缩放 + 噪声
                if np.random.rand() < 0.5:
                    shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1)
                    if shift > 0:
                        selected[:, :, :, shift:] = selected[:, :, :, :-shift].copy()
                        selected[:, :, :, :shift] = 0
                    elif shift < 0:
                        sa = abs(shift)
                        selected[:, :, :, :-sa] = selected[:, :, :, sa:].copy()
                        selected[:, :, :, -sa:] = 0

                    selected *= np.random.uniform(*SCALE_RANGE)
                    selected += np.random.normal(0, NOISE_STD, selected.shape).astype(np.float32)

        label = self.labels[real_idx]
        x = torch.from_numpy(selected)
        coord = torch.from_numpy(label["center_norm"])
        region = torch.tensor(label["region_idx"], dtype=torch.long)
        return x, coord, region


# ========================================
# 数据划分与 DataLoader
# ========================================

def build_splits(data_root=DATA_ROOT):
    """按区域分层划分 Train / Val / Test, 保证无数据泄露"""
    all_samples = build_sample_index(data_root)
    n = len(all_samples)
    print(f"[INFO] 共 {n} 个损伤样本")

    if n == 0:
        print("[WARN] 无真实数据, 使用 dummy 模式验证架构.")
        dummy = list(range(32))
        train_ds = NDTDataset(dummy[:24], "train", is_dummy=True)
        gs = train_ds.get_global_stats()
        val_ds = NDTDataset(dummy[24:28], "val", global_stats=gs, is_dummy=True)
        test_ds = NDTDataset(dummy[28:], "test", global_stats=gs, is_dummy=True)
        return train_ds, val_ds, test_ds

    region_groups = defaultdict(list)
    for i, (ri, *_) in enumerate(all_samples):
        region_groups[ri].append(i)

    np.random.seed(SEED)
    train_idx, val_idx, test_idx = [], [], []
    vt = 0
    for ri in sorted(region_groups):
        idxs = region_groups[ri]
        np.random.shuffle(idxs)
        if len(idxs) >= 2:
            train_idx.append(idxs[0])
            (val_idx if vt % 2 == 0 else test_idx).append(idxs[1])
            vt += 1
            train_idx.extend(idxs[2:])
        else:
            train_idx.extend(idxs)

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
