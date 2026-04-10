import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from config import SENSOR_COORDS, COORD_MIN, COORD_MAX

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(train_losses, val_losses, train_rmse, val_rmse, save_path=None):
    """训练/验证 Loss 与 RMSE 双面板曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', lw=2, label='Train Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, 'r--', lw=2, label='Val Loss')
    ax1.set(title='损失曲线', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(True, ls=':', alpha=0.6)

    ax2.plot(epochs, train_rmse, 'g-', lw=2, label='Train RMSE (mm)')
    if val_rmse:
        ax2.plot(epochs, val_rmse, 'm--', lw=2, label='Val RMSE (mm)')
    ax2.set(title='RMSE (mm)', xlabel='Epoch', ylabel='RMSE')
    ax2.legend(); ax2.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_localization_scatter(true_coords, pred_coords, log_vars=None,
                              accuracy=None, mean_std=None, save_path=None):
    """
    全景散点图: 275×275 mm 物理画板上
    红星 = GT,  蓝圈 = Pred,  虚线连接对应点对
    标题自动渲染 Accuracy 和 ±σ
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    board = COORD_MAX - COORD_MIN

    # 画板边界
    ax.add_patch(patches.Rectangle(
        (COORD_MIN, COORD_MIN), board, board,
        lw=2, ec='black', fc='none'))

    # 九宫格辅助线
    step = board / 3.0
    for k in range(1, 3):
        ax.axhline(COORD_MIN + k * step, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.axvline(COORD_MIN + k * step, color='gray', ls='--', lw=0.5, alpha=0.5)

    # 传感器
    ax.scatter(SENSOR_COORDS[:, 0], SENSOR_COORDS[:, 1],
               c='gray', marker='^', s=120, label='传感器', zorder=5)

    # GT + Pred
    for i in range(len(true_coords)):
        tx, ty = true_coords[i]
        px, py = pred_coords[i]
        ax.scatter(tx, ty, c='red', marker='*', s=180, zorder=10, alpha=0.9,
                   edgecolors='darkred', label='真实 (GT)' if i == 0 else '')
        ax.scatter(px, py, c='dodgerblue', marker='o', s=80, zorder=8, alpha=0.7,
                   edgecolors='navy', label='预测 (Pred)' if i == 0 else '')
        ax.plot([tx, px], [ty, py], 'k--', alpha=0.4, lw=1, zorder=6)

        if log_vars is not None:
            r = np.exp(0.5 * log_vars[i]) * 1.5
            ax.add_patch(patches.Circle((px, py), r, color='dodgerblue', alpha=0.1, zorder=1))

    ax.set_xlim(COORD_MIN - 20, COORD_MAX + 20)
    ax.set_ylim(COORD_MIN - 20, COORD_MAX + 20)
    ax.set_aspect('equal')

    title = "全景损伤定位评估"
    if accuracy is not None and mean_std is not None:
        title += f"\nAccuracy: {accuracy:.1f}% | ±σ: {mean_std:.2f} mm"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.legend(loc='upper right', framealpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_topology(edge_attns, bipartite_edges, save_path=None):
    """36 条二分图边的 Attention 拓扑热力图"""
    fig, ax = plt.subplots(figsize=(8, 8))

    G = nx.Graph()
    for idx, (x, y) in enumerate(SENSOR_COORDS):
        G.add_node(idx, pos=(x, y))
    pos = nx.get_node_attributes(G, 'pos')

    weights = np.array([float(edge_attns[i]) for i in range(len(bipartite_edges))])
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        norm_w = (weights - w_min) / (w_max - w_min)
    else:
        norm_w = np.ones_like(weights)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray',
                           node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    cmap = plt.cm.jet
    for i, (u, v) in enumerate(bipartite_edges):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=[cmap(norm_w[i])],
            width=norm_w[i] * 5 + 0.5,
            alpha=norm_w[i] * 0.8 + 0.1)

    ax.set_title('Bipartite Attention Topology (36 Edges)', fontsize=16)
    board = COORD_MAX - COORD_MIN
    ax.add_patch(patches.Rectangle(
        (COORD_MIN, COORD_MIN), board, board,
        lw=1.5, ec='black', fc='none', ls='--'))
    ax.axis('on')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
