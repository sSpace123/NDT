# GNN-SHM: 基于图神经网络和物理约束的损伤定位系统

> 适用于**变厚度扭转结构**中基于超声导波 (Lamb Wave) 的复杂损伤定位问题

## 项目简介

本项目实现了一套完整的结构健康监测 (Structural Health Monitoring, SHM) 算法流水线，核心创新点包括：

- **自适应物理信息 GNN**：基于可学习边权重的图神经网络，能够隐式补偿变厚度扭转结构中导波传播的曲率偏差
- **多任务联合训练**：同时进行区域分类和坐标回归（带不确定性估计）
- **物理约束损失**：通过 KL 散度软对齐飞行时间先验，实现柔性物理引导
- **鲁棒预处理**：DTW 对齐 → 带通滤波 → MAD 小波去噪 → Hilbert 包络 → CWT 时频变换

## 项目结构

```
gnn_shm/
├── config.py          # 全局物理常量与传感器坐标
├── dataset.py         # 信号预处理流水线与数据增强
├── model.py           # 自适应物理信息 GNN 网络架构
├── loss.py            # 多任务联合损失函数 (分类+回归+物理约束)
├── vis.py             # 论文级可视化 (训练曲线、定位散点、Attention拓扑)
├── train.py           # 训练/验证主入口脚本
├── requirements.txt   # Python 依赖
└── README.md          # 本文件
```

## 环境配置

### 方式一：使用 uv（推荐）

```bash
cd gnn_shm
uv venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 方式二：使用 pip

```bash
cd gnn_shm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 快速验证

使用合成数据快速验证整个训练流水线（无需真实数据）：

```bash
python train.py --demo --epochs 5
```

该命令会：
1. 生成 32 条合成样本
2. 运行 5 个 epoch 的训练/验证
3. 在 `outputs/` 目录生成训练曲线、定位散点图和 Attention 拓扑图

## 使用真实数据训练

1. 将 CSV 信号数据组织为 `(baseline_csv, damage_csv)` 路径对
2. 在 `train.py` 中替换数据集初始化代码：

```python
from dataset import SHMGraphDataset

sample_list = [
    (paths_dict, target_coord, target_cls),
    # paths_dict: {edge_idx: (base_csv_path, dmg_csv_path), ...}
    # target_coord: [x, y] in mm
    # target_cls: 0~8 区域标签
]
full_dataset = SHMGraphDataset(sample_list, augment=True)
```

3. 运行训练：

```bash
python train.py --epochs 200 --lr 1e-3 --batch_size 4
```

## 核心网络架构

```
输入 [B, 66, 4, 32, 2048]
        │
   ┌────┴────┐
   │ EdgeCNN │  2D CNN + SE → 每条路径的特征向量
   │ (×66)   │
   └────┬────┘
        │ [B, 66, 64]
   ┌────┴──────────┐
   │ SpatialGraph   │  可学习边权重 + Edge-to-Node 聚合
   │ (物理距离先验) │
   └────┬──────────┘
        │ [B, 12, 128]
   ┌────┴────────────┐
   │  Attention Pool  │  节点级 → 路径级 Attention
   └────┬────────────┘
        │ [B, 128]
   ┌────┼────┐
   │    │    │
  Cls  Reg  Attn
 [B,9] [B,3] [B,66]
```

## 消融实验指引

| 实验组 | 操作 |
|--------|------|
| 无自适应边权重 | `model.spatial_gnn.learnable_edge_weights.requires_grad_(False)` |
| 无物理约束 | `PINNLoss(lambda_phys=0)` |
| 无不确定性回归 | 将 `reg_head` 输出维度改为 2, 使用普通 MSE |

## 引用

如使用本代码，请引用相关工作。

## License

MIT
