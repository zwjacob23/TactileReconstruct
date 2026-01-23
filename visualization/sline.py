import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 1. 准备数据 (保持不变)
# ==========================================
try:
    s_ours = np.loadtxt('../output_pointclouds/test_2/predict/s_history_ours.txt')
    s_baseline = np.loadtxt('../output_pointclouds/test_2/predict/s_history_baseline.txt')
    epochs = np.arange(len(s_ours))
    print("成功加载真实数据文件。")
except:
    print("未找到文件，使用模拟数据演示效果...")
    epochs = np.arange(150)
    warmup = np.zeros(50)
    decay_base = -1.8 * (1 - np.exp(-0.04 * np.arange(100))) 
    s_baseline = np.concatenate([warmup, decay_base])
    decay_ours = -1.1 * (1 - np.exp(-0.06 * np.arange(100)))
    s_ours = np.concatenate([warmup, decay_ours])

# ==========================================
# 2. 全局绘图设置 (Times New Roman 风格)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # <--- 指定新罗马字体
    'mathtext.fontset': 'stix',         # <--- 数学公式也用 Times 风格
    'font.size': 12,                    # 字号稍微大一点点，Times 偏小
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'lines.linewidth': 2.0,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.fontsize': 10
})

# 创建画布
fig, ax = plt.subplots(figsize=(6, 4.5))

# ==========================================
# 3. 绘制核心曲线
# ==========================================
line_base, = ax.plot(epochs, s_baseline, color='#D62728', linestyle='--', 
                     label='Baseline (w/o PR-UW)')

line_ours, = ax.plot(epochs, s_ours, color='#1F77B4', linestyle='-', 
                     label='Ours (with PR-UW)')

# ==========================================
# 4. 辅助结构
# ==========================================
# A. 热身分界线
ax.axvline(x=50, color='k', linestyle='-', linewidth=0.8, alpha=0.7)

# B. 阴影区域
ax.fill_between(epochs, s_baseline, s_ours, where=(epochs>=50), 
                color='gray', alpha=0.2)

# ==========================================
# 5. 坐标轴与图例
# ==========================================
ax.set_xlabel('Training Epochs')
# 这里使用了 LaTeX 格式，配合 stix 字体，出来的 sigma 会很漂亮
ax.set_ylabel(r'Uncertainty Parameter $s$ ($\log \sigma^2$)')

ax.grid(True, linestyle='-')
ax.set_xlim(0, epochs[-1])

# 图例
fill_proxy = Patch(facecolor='gray', alpha=0.2, label='Regularization Effect')
ax.legend(handles=[line_base, line_ours, fill_proxy], loc='best')

plt.tight_layout()

# 保存
plt.savefig('s_evolution_times_new_roman.pdf', dpi=300, bbox_inches='tight')
plt.savefig('s_evolution_times_new_roman.png', dpi=300, bbox_inches='tight')

print("绘图完成！字体已设置为 Times New Roman。")
plt.show()