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
    FS, FC, WAVELET_NAME, WAVELET_LEVEL, TABULAR_DIM,
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
# 信号预处理与 CWT / TABULAR 特征提取
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
    提取图像与标量双模态特征:
    1. CWT 图 (用于 EdgeCNN) -> [4, 32, 2048]
    2. Tabular 手工特征 -> [5]
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
    base_exc = np.nan_to_num(base_df["excitation"].values[:min_len], nan=0.0).astype(np.float32)
    base_resp = np.nan_to_num(base_df["response"].values[:min_len], nan=0.0).astype(np.float32)
    dmg_exc = np.nan_to_num(dmg_df["excitation"].values[:min_len], nan=0.0).astype(np.float32)
    dmg_resp = np.nan_to_num(dmg_df["response"].values[:min_len], nan=0.0).astype(np.float32)

    diff_exc = dmg_exc - base_exc
    diff_resp = dmg_resp - base_resp

    diff_exc = _butter_bandpass(diff_exc, 0.5 * FC, 1.5 * FC, FS)
    diff_resp = _butter_bandpass(diff_resp, 0.5 * FC, 1.5 * FC, FS)

    diff_exc = _wavelet_denoise(diff_exc)
    diff_resp = _wavelet_denoise(diff_resp)

    env_exc = np.abs(hilbert(diff_exc))
    env_resp = np.abs(hilbert(diff_resp))

    # ---- 新增: 提取 5 维手工特征 (Tabular) ----
    try:
        cc = float(np.corrcoef(env_exc, env_resp)[0, 1]) if np.std(env_resp) > 0 else 0.0
        if np.isnan(cc): cc = 0.0
    except Exception:
        cc = 0.0
    tof = float(np.argmax(env_resp) - np.argmax(env_exc))
    tot_e = float(np.sum(diff_resp ** 2))
    env_e = float(np.sum(env_resp ** 2))
    hf_e = float(np.sum(np.abs(np.diff(diff_resp))))

    tab_feats = np.array([cc, tof, tot_e, env_e, hf_e], dtype=np.float32)
    tab_feats = np.nan_to_num(tab_feats, nan=0.0)

    # ---- 窗口裁剪: 用于 CWT ----
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

    return np.nan_to_num(tensor_cwt), tab_feats


# ========================================
# 数据目录扫描
# ========================================

def build_sample_index(data_root=DATA_ROOT):
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
# Dataset — 混合输入, 返回 (x_cwt, x_tab, coord_norm)
# ========================================

class NDTDataset(Dataset):
    def __init__(self, sample_index, mode="train", global_stats=None):
        self.samples = sample_index
        self.mode = mode
        self.repeat = AUGMENT_REPEAT if mode == "train" else 1

        self.data_cache = {}
        self.tab_cache = {}
        print(f"  [{mode}] {len(sample_index)} samples, extracting CWT & Tabular features...")
        for idx, (ri, tag, dmg_files, healthy_files) in enumerate(self.samples):
            cwt_list, tab_list = [], []
            for pi in BIPARTITE_INDICES:
                c, t = preprocess_pair(healthy_files[pi], dmg_files[pi])
                cwt_list.append(c)
                tab_list.append(t)
            self.data_cache[idx] = np.stack(cwt_list, axis=0) # [36, 4, 32, 2048]
            self.tab_cache[idx] = np.stack(tab_list, axis=0)  # [36, 5]

        # 独立进行 Z-score 归一化
        if global_stats is not None:
            self.g_mean_c, self.g_std_c, self.g_mean_t, self.g_std_t = global_stats
        elif mode == "train" and len(self.samples) > 0:
            all_c = np.stack(list(self.data_cache.values()))
            all_t = np.stack(list(self.tab_cache.values()))
            self.g_mean_c = np.mean(all_c, axis=(0, 1, 3, 4), keepdims=False)
            self.g_std_c = np.std(all_c, axis=(0, 1, 3, 4), keepdims=False) + 1e-8
            self.g_mean_t = np.mean(all_t, axis=(0, 1), keepdims=False)
            self.g_std_t = np.std(all_t, axis=(0, 1), keepdims=False) + 1e-8
        else:
            self.g_mean_c = np.zeros(IN_CHANNELS, dtype=np.float32)
            self.g_std_c = np.ones(IN_CHANNELS, dtype=np.float32)
            self.g_mean_t = np.zeros(TABULAR_DIM, dtype=np.float32)
            self.g_std_t = np.ones(TABULAR_DIM, dtype=np.float32)

        mc = self.g_mean_c[np.newaxis, :, np.newaxis, np.newaxis]
        sc = self.g_std_c[np.newaxis, :, np.newaxis, np.newaxis]
        mt = self.g_mean_t[np.newaxis, :]
        st = self.g_std_t[np.newaxis, :]
        
        for idx in range(len(self.samples)):
            n_c = (self.data_cache[idx] - mc) / sc
            n_t = (self.tab_cache[idx] - mt) / st
            self.data_cache[idx] = np.clip(n_c, -10, 10).astype(np.float32)
            self.tab_cache[idx] = np.clip(n_t, -10, 10).astype(np.float32)

        self.coords = []
        for ri, tag, _, _ in self.samples:
            self.coords.append(normalize_coord(DAMAGE_CENTERS[ri]))

    def get_global_stats(self):
        return (self.g_mean_c, self.g_std_c, self.g_mean_t, self.g_std_t)

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s_cwt = self.data_cache[real_idx].copy()
        s_tab = self.tab_cache[real_idx].copy()

        if self.mode == "train":
            # 1. 独立传感器边 Mask
            if np.random.rand() < 0.6:
                n_drop = np.random.randint(1, 6)
                drop_edges = np.random.choice(NUM_BIPARTITE_EDGES, n_drop, replace=False)
                s_cwt[drop_edges] = 0.0
                s_tab[drop_edges] = 0.0

            # 2. 受控时移 (仅影响 CWT 时序特征)
            if np.random.rand() < 0.7:
                shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1)
                if shift > 0:
                    s_cwt[:, :, :, shift:] = s_cwt[:, :, :, :-shift].copy()
                    s_cwt[:, :, :, :shift] = 0
                elif shift < 0:
                    sa = abs(shift)
                    s_cwt[:, :, :, :-sa] = s_cwt[:, :, :, sa:].copy()
                    s_cwt[:, :, :, -sa:] = 0

            # 3. 独立边缩放尺度扰动 (仅影响能量类特征)
            if np.random.rand() < 0.8:
                edge_scales = np.random.uniform(
                    SCALE_RANGE[0], SCALE_RANGE[1], size=(NUM_BIPARTITE_EDGES, 1)
                ).astype(np.float32)
                s_cwt *= edge_scales[:, :, np.newaxis, np.newaxis]
                # 只缩放能量列 (idx 2,3,4), 不影响互相关系数 (idx 0) 和 ToF (idx 1)
                s_tab[:, 2:] *= edge_scales

            # 4. 白噪声
            if np.random.rand() < 0.8:
                s_cwt += np.random.normal(0, NOISE_STD, s_cwt.shape).astype(np.float32)
                s_tab += np.random.normal(0, NOISE_STD, s_tab.shape).astype(np.float32)

        coord = torch.from_numpy(self.coords[real_idx].copy())
        return torch.from_numpy(s_cwt), torch.from_numpy(s_tab), coord


# ========================================
# 数据划分与 DataLoader
# ========================================

def build_splits(data_root=DATA_ROOT):
    all_samples = build_sample_index(data_root)
    n = len(all_samples)

    if n == 0:
        abs_root = os.path.abspath(data_root)
        raise RuntimeError(
            f"[FATAL] 在 '{abs_root}' 下未找到任何有效损伤样本!\n"
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
        x_cwt, x_tab, coord = ds[0]
        print(f"x_cwt={x_cwt.shape}, x_tab={x_tab.shape}, coord={coord.shape}")
        assert x_cwt.shape == (36, 4, 32, 2048)
        assert x_tab.shape == (36, TABULAR_DIM)
        assert coord.shape == (2,)
        print("Dataset shape test passed")
