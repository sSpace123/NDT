import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

from config import (
    DATA_ROOT, REGION_DIRS, DAMAGE_CENTERS,
    NUM_PAIRS, CSV_HEADER_LINES,
    WINDOW_HALF_SIZE, IN_CHANNELS, BATCH_SIZE, SEED, normalize_coord,
    AUGMENT_REPEAT, SCALE_RANGE, TIME_SHIFT_MAX,
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
# DSP 信号处理与特征提取
# ========================================

def _butter_bandpass(data, lowcut, highcut, fs, order=4):
    """四阶 Butterworth 带通滤波"""
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)


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


def extract_physical_features(baseline_sig, damaged_sig):
    """
    独立内聚的 DSP 收敛特征提取
    提取 5 维物理敏感手工特征:
      (1) 互相关系数: baseline_sig 与 damaged_sig 的最大归一化互相关系数
      (2) 包络 ToF: 差分信号的 Hilbert 包络的最大值出现时间点 (索引)
      (3) 信号总能量: 差分信号的平方和
      (4) 包络能量: 差分信号 Hilbert 包络的平方和
      (5) 小波高频能量: db4 单层小波分解后细节系数(cD)平方和
    """
    # 防御性对齐
    min_len = min(len(baseline_sig), len(damaged_sig))
    base = baseline_sig[:min_len]
    dmg = damaged_sig[:min_len]
    diff_sig = dmg - base

    # 1. 最大归一化互相关系数
    norm_factor = np.linalg.norm(base) * np.linalg.norm(dmg)
    if norm_factor > 1e-12:
        # np.corrcoef 也可以直接作为均值标准化的相似度,
        # 这里严格根据信号内积使用 np.correlate 求滑动满相关最大值
        xcorr = signal.correlate(dmg, base, mode='same')
        cc = np.max(xcorr) / norm_factor
    else:
        cc = 0.0

    # 2. 信号总能量
    tot_e = float(np.sum(diff_sig ** 2))

    # 3 & 4. Hilbert 包络计算 ToF 和包络能量
    # 包络异常防护
    if np.max(np.abs(diff_sig)) < 1e-12:
        tof = 0.0
        env_e = 0.0
    else:
        env = np.abs(signal.hilbert(diff_sig))
        tof = float(np.argmax(env))
        env_e = float(np.sum(env ** 2))

    # 5. 小波高频细节能量
    try:
        cA, cD = pywt.dwt(diff_sig, 'db4')
        hf_e = float(np.sum(cD ** 2))
    except Exception:
        hf_e = 0.0

    feats = np.array([cc, tof, tot_e, env_e, hf_e], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0)


def preprocess_pair(healthy_csv, damage_csv):
    """
    单路径预处理管道：
    加载 -> 滤波去噪 -> (提 5 维 DSP 特征) -> 截断 2048 长度 -> (提 CWT 画像)
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

    # 计算 5 维物理敏感手工特征 (在全长信号与滤波前直接计算，最真实的保留原始物理意义)
    tabular_feats = extract_physical_features(base_resp, dmg_resp)

    # 开始准备 CWT
    diff_exc = dmg_exc - base_exc
    diff_resp = dmg_resp - base_resp

    diff_exc = _butter_bandpass(diff_exc, 0.5 * FC, 1.5 * FC, FS)
    diff_resp = _butter_bandpass(diff_resp, 0.5 * FC, 1.5 * FC, FS)

    diff_exc = _wavelet_denoise(diff_exc)
    diff_resp = _wavelet_denoise(diff_resp)

    env_exc = np.abs(signal.hilbert(diff_exc))
    env_resp = np.abs(signal.hilbert(diff_resp))

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

    return np.nan_to_num(tensor_cwt), tabular_feats


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
# Dataset — 预处理内存缓存及 Z-Score 全局防泄露
# ========================================

class NDTDataset(Dataset):
    def __init__(self, sample_index, mode="train", global_stats=None):
        self.samples = sample_index
        self.mode = mode
        self.repeat = AUGMENT_REPEAT if mode == "train" else 1

        self.data_cache = {}                   # 缓存 CWT 特征 (归一化后)
        self.raw_tab_cache = {}                 # 缓存 Tabular 原始特征 (增强前)
        self.tab_cache = {}                     # 缓存 Tabular 特征 (归一化后, 非 train 用)

        print(f"  [{mode}] {len(sample_index)} samples, extracting CWT & Tabular features...")
        for idx, (ri, tag, dmg_files, healthy_files) in enumerate(self.samples):
            cwt_list, tab_list = [], []
            for pi in BIPARTITE_INDICES:
                c, t = preprocess_pair(healthy_files[pi], dmg_files[pi])
                cwt_list.append(c)
                tab_list.append(t)
            self.data_cache[idx] = np.stack(cwt_list, axis=0)    # [36, 4, 32, 2048]
            self.raw_tab_cache[idx] = np.stack(tab_list, axis=0) # [36, 5] 原始物理值

        # -------------------------------------------------------------
        # Z-score 归一化统计量 (防数据泄露)
        # Val/Test 强制使用 Train 的统计量
        # -------------------------------------------------------------
        if global_stats is not None:
            self.g_mean_c, self.g_std_c, self.g_mean_t, self.g_std_t = global_stats
        elif mode == "train" and len(self.samples) > 0:
            all_c = np.stack(list(self.data_cache.values()))
            all_t = np.stack(list(self.raw_tab_cache.values()))
            self.g_mean_c = np.mean(all_c, axis=(0, 1, 3, 4), keepdims=False)
            self.g_std_c = np.std(all_c, axis=(0, 1, 3, 4), keepdims=False) + 1e-8
            self.g_mean_t = np.mean(all_t, axis=(0, 1), keepdims=False)
            self.g_std_t = np.std(all_t, axis=(0, 1), keepdims=False) + 1e-8
        else:
            self.g_mean_c = np.zeros(IN_CHANNELS, dtype=np.float32)
            self.g_std_c = np.ones(IN_CHANNELS, dtype=np.float32)
            self.g_mean_t = np.zeros(TABULAR_DIM, dtype=np.float32)
            self.g_std_t = np.ones(TABULAR_DIM, dtype=np.float32)

        # CWT 归一化 (直接覆写缓存, CWT 增强只做空间操作不影响统计量)
        mc = self.g_mean_c[np.newaxis, :, np.newaxis, np.newaxis]
        sc = self.g_std_c[np.newaxis, :, np.newaxis, np.newaxis]
        for idx in range(len(self.samples)):
            n_c = (self.data_cache[idx] - mc) / sc
            self.data_cache[idx] = np.clip(n_c, -10, 10).astype(np.float32)

        # Tabular: train 模式保留原始值用于增强后再归一化;
        # val/test 模式直接归一化缓存
        if mode != "train":
            mt = self.g_mean_t[np.newaxis, :]
            st = self.g_std_t[np.newaxis, :]
            for idx in range(len(self.samples)):
                n_t = (self.raw_tab_cache[idx] - mt) / st
                self.tab_cache[idx] = np.clip(n_t, -10, 10).astype(np.float32)

        # 回归标签
        self.coords = []
        for ri, tag, _, _ in self.samples:
            self.coords.append(normalize_coord(DAMAGE_CENTERS[ri]))

    def get_global_stats(self):
        return (self.g_mean_c, self.g_std_c, self.g_mean_t, self.g_std_t)

    def __len__(self):
        return len(self.samples) * self.repeat

    def _normalize_tab(self, raw_tab):
        """对原始 tabular 特征做 Z-score 归一化"""
        mt = self.g_mean_t[np.newaxis, :]
        st = self.g_std_t[np.newaxis, :]
        return np.clip((raw_tab - mt) / st, -10, 10).astype(np.float32)

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s_cwt = self.data_cache[real_idx].copy()

        if self.mode == "train":
            # 取原始物理值的副本, 增强在归一化前进行
            s_tab_raw = self.raw_tab_cache[real_idx].copy()

            # --- 1. Edge Dropout: 70% 概率随机遮断 3~8 条边 ---
            if np.random.rand() < 0.7:
                n_drop = np.random.randint(3, 9)
                drop_edges = np.random.choice(NUM_BIPARTITE_EDGES, n_drop, replace=False)
                s_cwt[drop_edges] = 0.0
                s_tab_raw[drop_edges] = 0.0

            # --- 2. CWT 时序平移 (±5 点) ---
            if np.random.rand() < 0.7:
                shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1)
                if shift > 0:
                    s_cwt[:, :, :, shift:] = s_cwt[:, :, :, :-shift].copy()
                    s_cwt[:, :, :, :shift] = 0
                elif shift < 0:
                    sa = abs(shift)
                    s_cwt[:, :, :, :-sa] = s_cwt[:, :, :, sa:].copy()
                    s_cwt[:, :, :, -sa:] = 0

            # --- 3. Tabular 物理微扰 (在原始物理值上操作) ---
            # ToF 抖动: ±2 个采样点 (物理单位, 非标准差)
            if np.random.rand() < 0.8:
                s_tab_raw[:, 1] += np.random.uniform(-2.0, 2.0, size=(NUM_BIPARTITE_EDGES,))

            # 能量缩放: 乘以 0.9~1.1 (在原始幅值上才有物理意义)
            if np.random.rand() < 0.8:
                energy_scale = np.random.uniform(0.9, 1.1, size=(NUM_BIPARTITE_EDGES, 1))
                s_tab_raw[:, 2:5] *= energy_scale
                # CWT 同步缩放保持多模态一致性
                s_cwt *= energy_scale[:, :, np.newaxis, np.newaxis]

            # 增强完毕后再做 Z-score 归一化
            s_tab = self._normalize_tab(s_tab_raw)
        else:
            # Val/Test: 直接读取预归一化的缓存
            s_tab = self.tab_cache[real_idx].copy()

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
