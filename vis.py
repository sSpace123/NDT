import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from config import SENSOR_COORDS, COORD_MIN, COORD_MAX, NUM_REGIONS, REGION_CENTERS, REGION_NAMES

# ========================================
# 六、 实验可视化与论文作图 
# ========================================

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

def plot_training_curves(train_losses, val_losses, train_rmse, val_rmse, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Total Loss', linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r--', label='Val Total Loss', linewidth=2)
    axes[0].set_title('训练与验证损失曲线', fontsize=16)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].plot(epochs, train_rmse, 'g-', label='Train RMSE (mm)', linewidth=2)
    if val_rmse:
        axes[1].plot(epochs, val_rmse, 'm--', label='Val RMSE (mm)', linewidth=2)
    axes[1].set_title('坐标回归均方根误差 (RMSE)', fontsize=16)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('RMSE (mm)', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_localization_scatter(true_coords, pred_coords, log_vars=None, accuracy=None, mean_std=None, save_path=None):
    """
    全景散点对比图: 
    - 画出板的 275x275 边界包含所有结果
    - true_coords 红色星形，pred_coords 蓝色圆圈，加半透明虚线
    - legend 中显示 Acc 和 confidence (mean_std)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    width = COORD_MAX - COORD_MIN
    rect = patches.Rectangle((COORD_MIN, COORD_MIN), width, width, linewidth=2, edgecolor='black', facecolor='none', linestyle='-')
    ax.add_patch(rect)
    
    # 画区域格子划分 (3x3 九宫格辅助线)
    step = width / 3.0
    for i in range(1, 4):
        ax.axhline(COORD_MIN + i * step, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(COORD_MIN + i * step, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
    ax.scatter(SENSOR_COORDS[:, 0], SENSOR_COORDS[:, 1], c='gray', marker='^', s=120, label='传感器', zorder=5)
    
    # 绘制
    for i in range(len(true_coords)):
        tx, ty = true_coords[i]
        px, py = pred_coords[i]
        
        label_t = '真实坐标 (GT)' if i == 0 else ""
        ax.scatter(tx, ty, c='red', marker='*', s=180, zorder=10, label=label_t, alpha=0.9, edgecolors='darkred')
        
        label_p = '预测坐标 (Pred)' if i == 0 else ""
        ax.scatter(px, py, c='dodgerblue', marker='o', s=80, zorder=8, label=label_p, alpha=0.7, edgecolors='navy')
        
        ax.plot([tx, px], [ty, py], 'k--', alpha=0.4, linewidth=1.0, zorder=6)
        
        if log_vars is not None:
            std = np.exp(0.5 * log_vars[i]) 
            circle = patches.Circle((px, py), radius=std * 1.5, color='dodgerblue', alpha=0.1, zorder=1)
            ax.add_patch(circle)
            
    ax.set_xlim(COORD_MIN - 20, COORD_MAX + 20)
    ax.set_ylim(COORD_MIN - 20, COORD_MAX + 20)
    
    title_str = "全景损伤定位评估结果"
    if accuracy is not None and mean_std is not None:
        title_str += f"\n区域分类 Accuracy: {accuracy:.1f}% | 平均定位误差带(均布预估): ±{mean_std:.2f} mm"
        
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X 轴 (mm)', fontsize=14)
    ax.set_ylabel('Y 轴 (mm)', fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(False)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_topology(edge_attns, bipartite_edges, save_path=None):
    """
    36 条边图 Attention 拓扑
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    G = nx.Graph()
    for idx, (x, y) in enumerate(SENSOR_COORDS):
        G.add_node(idx, pos=(x, y))
        
    pos = nx.get_node_attributes(G, 'pos')
    
    edges_info = []
    for i, (u, v) in enumerate(bipartite_edges):
        w = float(edge_attns[i])
        edges_info.append((u, v, w))
            
    weights = np.array([col[2] for col in edges_info])
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights = (weights - w_min) / (w_max - w_min)
        
    cmap = plt.cm.jet 
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    for i, (u, v, _) in enumerate(edges_info):
        alpha = float(weights[i]) * 0.8 + 0.1 
        width = float(weights[i]) * 5.0 + 0.5 
        color = cmap(weights[i])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=[color], width=width, alpha=alpha)
        
    ax.set_title('36-Edge Bipartite Attention Topology', fontsize=16)
    
    width_box = COORD_MAX - COORD_MIN
    rect = patches.Rectangle((COORD_MIN, COORD_MIN), width_box, width_box, linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.axis('on')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
