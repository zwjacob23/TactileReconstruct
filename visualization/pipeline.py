import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# 全局设置：大字体，适合缩小后的示意图
plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif'})

def draw_revised_schematics():
    # 1. 模拟数据 (保持之前的逻辑)
    x = np.linspace(0, 100, 300)

    # --- 辅助数据 (Auxiliary Tasks) ---
    # 两个配角，动态变化
    y_rad = 0.6 / (1 + np.exp(-0.1 * (x - 10))) + 0.3
    y_theta = 0.5 + 0.1 * np.sin(x * 0.15)
    # 平滑 + 噪声
    y_rad = gaussian_filter1d(y_rad + np.random.normal(0, 0.02, len(x)), sigma=3)
    y_theta = gaussian_filter1d(y_theta + np.random.normal(0, 0.02, len(x)), sigma=3)

    # --- 关键数据 1: 权重坍塌 (Collapse - Baseline) ---
    y_collapse = 2.2 * np.exp(-0.025 * x) + 0.4
    y_collapse = gaussian_filter1d(y_collapse + np.random.normal(0, 0.04, len(x)), sigma=3)

    # --- 关键数据 2: 权重恢复 (Recovery - Ours) ---
    y_recover = 1.8 * np.exp(-0.1 * x) + 2.0 / (1 + np.exp(-0.15 * (x - 45)))
    y_recover = gaussian_filter1d(y_recover + np.random.normal(0, 0.04, len(x)), sigma=3)

    # ==========================================
    # 图 1: Dynamic Weighting (Baseline)
    # 需求：辅助任务要有颜色，展示动态性；图注说明是权重即可。
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 3.2)) # 稍微增加一点高度放图例
    
    # 配角 (使用鲜明的颜色: 橙色、绿色)
    ax1.plot(x, y_rad, color='#ff7f0e', lw=3, ls='--', label=r'$\lambda_{Aux1}$ (Radius)')
    ax1.plot(x, y_theta, color='#2ca02c', lw=3, ls=':', label=r'$\lambda_{Aux2}$ (Angle)')

    # 主角 (蓝色实线 - 展示基线的情况)
    ax1.plot(x, y_collapse, color='#1f77b4', lw=5, label=r'$\lambda_{PC}$ (Main Task)')

    # 标注
    ax1.set_title("Dynamic Uncertainty Weighting", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Task Weight ($\lambda$)", fontsize=16)
    ax1.set_xlabel("Training Progress", fontsize=16)
    ax1.set_xticks([]); ax1.set_yticks([]) # 去刻度
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 3.0)
    
    # 去边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 图例 (放在显眼位置，展示三个都在变)
    ax1.legend(loc='upper right', fontsize=13, frameon=False)

    plt.tight_layout()
    plt.savefig('plot1_dynamic_baseline.png', dpi=300)
    print("生成完成: plot1_dynamic_baseline.png")


    # ==========================================
    # 图 2: Effect of PR Loss (Comparison)
    # 需求：辅助任务变灰；重点对比蓝色虚线和实线。
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 3.2))

    # 配角 (变成灰色、半透明，退居背景)
    ax2.plot(x, y_rad, color='gray', lw=2, ls='--', alpha=0.3)
    ax2.plot(x, y_theta, color='gray', lw=2, ls=':', alpha=0.3)
    # 在图2里不需要给配角加图例了，因为它们不重要

    # 前任主角 (蓝色虚线、半透明 - 对照组)
    ax2.plot(x, y_collapse, color='#1f77b4', lw=3, ls='--', alpha=0.5, label=r'$\lambda_{PC}$ (w/o PR Loss)')

    # 现任主角 (蓝色实线、高亮 - 实验组)
    ax2.plot(x, y_recover, color='#1f77b4', lw=5, label=r'$\lambda_{PC}$ (w/ PR Loss)')

    # 标注
    ax2.set_title("Effect of PR Loss Regularization", fontsize=20, fontweight='bold')
    ax2.set_ylabel("Weight ($\lambda$)", fontsize=16)
    ax2.set_xlabel("Training Progress", fontsize=16)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 3.0)
    
    # 去边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 关键标注: 扭转趋势
    ax2.annotate('Recovery', 
                 xy=(50, 1.2), xytext=(65, 0.6),
                 color='#1f77b4', fontsize=16, fontweight='bold',
                 arrowprops=dict(facecolor='#1f77b4', shrink=0.05))

    # 图例 (重点区分实线和虚线)
    ax2.legend(loc='upper center', fontsize=13, frameon=False, ncol=2) # ncol=2横向排布

    plt.tight_layout()
    plt.savefig('plot2_prloss_effect.png', dpi=300)
    print("生成完成: plot2_prloss_effect.png")
    
    plt.show()

if __name__ == "__main__":
    draw_revised_schematics()