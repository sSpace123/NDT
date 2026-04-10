import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from config import SENSOR_COORDS, COORD_MIN, COORD_MAX

# ========================================
# 六、 实验可视化与论文作图 
# ========================================

# 全局字体配置，解决 Matplotlib 中文乱码和负号显示问题 (支持 Windows 与 Linux)
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题

def plot_training_curves(train_losses, val_losses, train_rmse, val_rmse, save_path=None):
    """
    绘制 Train/Val 的 Loss 曲线和回归坐标误差 (RMSE) 曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- 画 Loss 曲线 ---
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Total Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r--', label='Val Total Loss', linewidth=2)
    axes[0].set_title('训练与验证损失曲线', fontsize=16)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # --- 画 RMSE 曲线 ---
    axes[1].plot(epochs, train_rmse, 'g-', label='Train RMSE (mm)', linewidth=2)
    axes[1].plot(epochs, val_rmse, 'm--', label='Val RMSE (mm)', linewidth=2)
    axes[1].set_title('坐标回归均方根误差 (RMSE)', fontsize=16)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('RMSE (mm)', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线保存至: {save_path}")
    plt.show()

def plot_localization_scatter(true_coords, pred_coords, log_vars=None, save_path=None):
    """
    定位散点对比图: 二维坐标系中，画出板的边界框，绘制真实坐标与预测坐标的对比。
    若传入 log_vars(不确定度回归结果)，则画出对应的误差/置信椭圆。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. 绘制板的边界框
    width = COORD_MAX - COORD_MIN
    rect = patches.Rectangle((COORD_MIN, COORD_MIN), width, width, linewidth=2, edgecolor='black', facecolor='none', linestyle='-')
    ax.add_patch(rect)
    
    # 2. 绘制传感器节点
    ax.scatter(SENSOR_COORDS[:, 0], SENSOR_COORDS[:, 1], c='gray', marker='^', s=100, label='传感器位置')
    
    # 3. 绘制预测结果与真值
    for i in range(len(true_coords)):
        tx, ty = true_coords[i]
        px, py = pred_coords[i]
        
        # 真实位置 (星号)
        label_t = '真实坐标 (True)' if i == 0 else ""
        ax.scatter(tx, ty, c='red', marker='*', s=150, zorder=5, label=label_t)
        
        # 预测位置 (圆圈)
        label_p = '预测坐标 (Pred)' if i == 0 else ""
        ax.scatter(px, py, c='blue', marker='o', s=80, zorder=4, label=label_p)
        
        # 绘制残差连接线
        ax.plot([tx, px], [ty, py], 'k--', alpha=0.3)
        
        # 绘制置信椭圆 (假设 log_var 表示二维协方差等向分布的高斯对数方差)
        if log_vars is not None:
            std = np.exp(0.5 * log_vars[i]) # 还原至标准差并适当放大为圆圈
            circle = patches.Circle((px, py), radius=std * 2.0, color='blue', alpha=0.15)
            ax.add_patch(circle)
            
    ax.set_xlim(COORD_MIN - 20, COORD_MAX + 20)
    ax.set_ylim(COORD_MIN - 20, COORD_MAX + 20)
    ax.set_title('结构表面损伤定位与不确定度展示', fontsize=16)
    ax.set_xlabel('X 轴位置 (mm)', fontsize=14)
    ax.set_ylabel('Y 轴位置 (mm)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle="-.", alpha=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_attention_topology(edge_attns, save_path=None):
    """
    传感器拓扑 Attention 热力图: 
    极大展示网络自行寻找的对损伤最具有敏感判断力的波流路径。
    :param edge_attns: 一维数组，长度 66，值为网络输出的归一化注意力权重
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    G = nx.Graph()
    # 根据 SENSOR_COORDS 画出 12 个节点
    for idx, (x, y) in enumerate(SENSOR_COORDS):
        G.add_node(idx, pos=(x, y))
        
    pos = nx.get_node_attributes(G, 'pos')
    
    # 重建边信息与提取极值
    edges_info = []
    idx = 0
    for i in range(12):
        for j in range(i+1, 12):
            w = float(edge_attns[idx])
            edges_info.append((i, j, w))
            idx += 1
            
    # Normalize weights to range [0, 1] for visual thickness 
    weights = np.array([col[2] for col in edges_info])
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights = (weights - w_min) / (w_max - w_min)
        
    # 根据权重值过滤并画图，设置线条粗细与颜色映射
    # Colormap: 深红代表高 Attention，浅黄/灰代表低 Attention
    cmap = plt.cm.jet 
    
    # 画节点
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    # 画边
    for i, (u, v, _) in enumerate(edges_info):
        alpha = float(weights[i]) * 0.8 + 0.1 # 最小可见度为 0.1
        width = float(weights[i]) * 5.0 + 0.5 # 线条厚度范围 0.5 ~ 5.5
        color = cmap(weights[i])
        
        # 只突出表现 Attention 排名前列的边，或全体绘制(调参呈现)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=[color], width=width, alpha=alpha)
        
    ax.set_title('导波网络物理路径 Attention 权重感知拓扑图', fontsize=16)
    
    # 模拟边界框辅助线
    width_box = COORD_MAX - COORD_MIN
    rect = patches.Rectangle((COORD_MIN, COORD_MIN), width_box, width_box, linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.axis('on')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 测试函数能否正常生成随机伪数据并展示制图
    dummy_attns = np.random.rand(66)
    plot_attention_topology(dummy_attns)
    
    dummy_true = np.array([[0, 0], [50, -50], [-100, 100]])
    dummy_pred = dummy_true + np.random.normal(0, 10, (3, 2))
    dummy_vars = np.random.uniform(-1, 1, 3) 
    plot_localization_scatter(dummy_true, dummy_pred, dummy_vars)
