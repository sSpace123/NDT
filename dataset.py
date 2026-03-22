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
