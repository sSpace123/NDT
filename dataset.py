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
    DATA_ROOT, REGION_DIRS, DAMAGE_CENTERS,
    NUM_PAIRS, CSV_HEADER_LINES,
    WINDOW_HALF_SIZE, IN_CHANNELS, BATCH_SIZE, SEED, normalize_coord,
    AUGMENT_REPEAT, NOISE_STD, SCALE_RANGE, TIME_SHIFT_MAX,
    FS, FC, WAVELET_NAME, WAVELET_LEVEL,
)

# ========================================
# 预计算 36 边二分图索引映射 (Left 0~5 × Right 6~11)
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
    单条路径完整 Pipeline: 读取 → 差分 → 滤波 → 去噪 → 包络 → 窗口裁剪 → CWT
    返回: [4, 32, 2048]
    """
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

    for col in ["excitation", "response"]:
        base_df[col] = pd.to_numeric(base_df[col], errors='coerce')
        dmg_df[col] = pd.to_numeric(dmg_df[col], errors='coerce')

    min_len = min(len(base_df), len(dmg_df))
    base_exc = np.nan_to_num(base_df["excitation"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    base_resp = np.nan_to_num(base_df["response"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmg_exc = np.nan_to_num(dmg_df["excitation"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmg_resp = np.nan_to_num(dmg_df["response"].values[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    diff_exc = dmg_exc - base_exc
    diff_resp = dmg_resp - base_resp

    diff_exc = _butter_bandpass(diff_exc, 0.5 * FC, 1.5 * FC, FS)
    diff_resp = _butter_bandpass(diff_resp, 0.5 * FC, 1.5 * FC, FS)

    diff_exc = _wavelet_denoise(diff_exc)
    diff_resp = _wavelet_denoise(diff_resp)

    env_exc = np.abs(hilbert(diff_exc))
    env_resp = np.abs(hilbert(diff_resp))

    # 物理基准: 以激励包络对齐, 保留不同路径的相对飞行时间差 (ToF)
    peak = np.argmax(env_exc)
    length = len(env_exc)
    start = max(0, peak - WINDOW_HALF_SIZE)
    end = min(length, peak + WINDOW_HALF_SIZE)
    pad_l = max(0, WINDOW_HALF_SIZE - peak)
    pad_r = max(0, (peak + WINDOW_HALF_SIZE) - length)

    def _crop_pad(sig):
        return np.pad(sig[start:end], (pad_l, pad_r), 'constant')

    tensor_cwt = np.stack([
        _get_cwt(_crop_pad(diff_exc)),
        _get_cwt(_crop_pad(diff_resp)),
        _get_cwt(_crop_pad(env_exc)),
        _get_cwt(_crop_pad(env_resp)),
    ], axis=0).astype(np.float32)

    return np.nan_to_num(tensor_cwt, nan=0.0, posinf=0.0, neginf=0.0)


# ========================================
# 数据目录扫描
# ========================================

def build_sample_index(data_root=DATA_ROOT):
    """扫描子目录, 返回 [(region_idx, tag, dmg_files, healthy_files), ...]"""
    abs_root = os.path.abspath(data_root)
    if not os.path.exists(abs_root):
        raise RuntimeError(f"[FATAL] DATA_ROOT 不存在: '{abs_root}'")

    samples = []
    for ri, rd in enumerate(REGION_DIRS):
        dp = os.path.join(abs_root, rd)
        if not os.path.isdir(dp):
            continue
        tags = set()
        for f in glob.glob(os.path.join(dp, "tek*ALL*.csv")):
            m = re.search(r"ALL(\w+)\.csv", os.path.basename(f))
            if m:
                tags.add(m.group(1))
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
# Dataset — 纯回归, 返回 (x, coord_norm) 无 region_idx
# ========================================

class NDTDataset(Dataset):
    """
    纯回归数据集: 返回 (x, coord_norm).
    - x: [36, 4, 32, 2048]  二分图边 CWT 特征
    - coord_norm: [2]  归一化坐标 [-1, 1]
    """

    def __init__(self, sample_index, mode="train", global_stats=None):
        self.samples = sample_index
        self.mode = mode
        self.repeat = AUGMENT_REPEAT if mode == "train" else 1

        # 预处理并缓存
        self.data_cache = {}
        print(f"  [{mode}] {len(sample_index)} samples, extracting {NUM_BIPARTITE_EDGES}-edge CWT...")
        for idx, (ri, tag, dmg_files, healthy_files) in enumerate(self.samples):
            signals = np.stack([
                preprocess_pair(healthy_files[pi], dmg_files[pi])
                for pi in BIPARTITE_INDICES
            ], axis=0)
            self.data_cache[idx] = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        # 全局 Z-score 归一化
        if global_stats is not None:
            self.global_mean, self.global_std = global_stats
        elif mode == "train" and len(self.samples) > 0:
            all_data = np.stack(list(self.data_cache.values()))
            self.global_mean = np.mean(all_data, axis=(0, 1, 3, 4), keepdims=False)
            self.global_std = np.std(all_data, axis=(0, 1, 3, 4), keepdims=False) + 1e-8
        else:
            self.global_mean = np.zeros(IN_CHANNELS, dtype=np.float32)
            self.global_std = np.ones(IN_CHANNELS, dtype=np.float32)

        mean = self.global_mean[np.newaxis, :, np.newaxis, np.newaxis]
        std = self.global_std[np.newaxis, :, np.newaxis, np.newaxis]
        for idx in range(len(self.samples)):
            normed = (self.data_cache[idx] - mean) / std
            self.data_cache[idx] = np.clip(normed, -10, 10).astype(np.float32)

        # 纯坐标标签 (无 region_idx)
        self.coords = []
        for ri, tag, _, _ in self.samples:
            self.coords.append(normalize_coord(DAMAGE_CENTERS[ri]))

    def get_global_stats(self):
        return (self.global_mean, self.global_std)

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        selected = self.data_cache[real_idx].copy()

        if self.mode == "train":
            # 1. Sensor Dropout (p=0.6): mask 1~5 条边
            if np.random.rand() < 0.6:
                n_drop = np.random.randint(1, 6)
                drop_edges = np.random.choice(
                    NUM_BIPARTITE_EDGES, n_drop, replace=False)
                selected[drop_edges] = 0.0

            # 2. 受控时移 (p=0.7): ±5 点
            if np.random.rand() < 0.7:
                shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1)
                if shift > 0:
                    selected[:, :, :, shift:] = selected[:, :, :, :-shift].copy()
                    selected[:, :, :, :shift] = 0
                elif shift < 0:
                    sa = abs(shift)
                    selected[:, :, :, :-sa] = selected[:, :, :, sa:].copy()
                    selected[:, :, :, -sa:] = 0

            # 3. 独立边缩放 (p=0.8): 36 条边各自独立
            if np.random.rand() < 0.8:
                edge_scales = np.random.uniform(
                    SCALE_RANGE[0], SCALE_RANGE[1],
                    size=(NUM_BIPARTITE_EDGES, 1, 1, 1)
                ).astype(np.float32)
                selected *= edge_scales

            # 4. AWGN (p=0.8)
            if np.random.rand() < 0.8:
                selected += np.random.normal(
                    0, NOISE_STD, selected.shape).astype(np.float32)

        x = torch.from_numpy(selected)
        coord = torch.from_numpy(self.coords[real_idx].copy())
        return x, coord


# ========================================
# 数据划分与 DataLoader
# ========================================

def build_splits(data_root=DATA_ROOT):
    """按区域分层划分 Train / Val / Test"""
    all_samples = build_sample_index(data_root)
    n = len(all_samples)

    if n == 0:
        abs_root = os.path.abspath(data_root)
        raise RuntimeError(
            f"[FATAL] 在 '{abs_root}' 下未找到任何有效损伤样本!\n"
            f"  期望的子目录: {REGION_DIRS}\n"
            f"  期望的文件格式: tek{{0000~0065}}ALL{{tag}}.csv"
        )

    print(f"[INFO] {n} damage samples found")

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

    assert not (set(train_idx) & set(val_idx)), "Train/Val overlap!"
    assert not (set(train_idx) & set(test_idx)), "Train/Test overlap!"
    assert not (set(val_idx) & set(test_idx)), "Val/Test overlap!"

    def _desc(indices):
        return [f"{REGION_DIRS[all_samples[i][0]]}({all_samples[i][1]})" for i in indices]

    print(f"  Train({len(train_idx)}): {_desc(train_idx)}")
    print(f"  Val  ({len(val_idx)}):  {_desc(val_idx)}")
    print(f"  Test ({len(test_idx)}):  {_desc(test_idx)}")

    train_ds = NDTDataset([all_samples[i] for i in train_idx], "train")
    gs = train_ds.get_global_stats()
    val_ds = NDTDataset([all_samples[i] for i in val_idx], "val", global_stats=gs)
    test_ds = NDTDataset([all_samples[i] for i in test_idx], "test", global_stats=gs)
    return train_ds, val_ds, test_ds


def get_dataloaders(data_root=DATA_ROOT):
    train_ds, val_ds, test_ds = build_splits(data_root)
    # num_workers=0: 数据已预缓存在内存, 多进程无加速且 Windows 上易死锁
    import torch
    pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=pin),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin),
        DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=pin),
    )


if __name__ == '__main__':
    samples = build_sample_index()
    print(f"Found {len(samples)} samples")
    if samples:
        ds = NDTDataset(samples[:1], mode="test")
        x, coord = ds[0]
        print(f"x={x.shape}, coord={coord.shape}")
        assert x.shape == (36, 4, 32, 2048)
        assert coord.shape == (2,)
        print("Dataset shape test passed")

