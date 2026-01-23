import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小，方便论文阅读
plt.rcParams.update({'font.size': 14}) 

# 准备数据
x = np.linspace(0.1, 2.5, 100)  # s 的范围

# 模拟函数形状
# 1. 任务损失 (模拟 decay): L_task ~ 1/s
y_task_weighted = 1.0 / (x + 0.2) 

# 2. 正则项 (模拟 Soft Anchor): Reg ~ s^2
y_prior = 0.4 * x**2 

# 3. 总损失: Total = Task + Reg
y_total = y_task_weighted + y_prior

# 找到最小值的点 (用于画那个绿色的点)
min_idx = np.argmin(y_total)
x_min = x[min_idx]
y_min = y_total[min_idx]

# 创建画布：1行2列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# ==========================================
# 左图：Standard Uncertainty Weighting
# ==========================================
# 为了模拟左图 "Optimization Collapse"，我们只画一个单调下降的曲线
# 这里用 y_task_weighted + 一个很小的线性项，或者直接画 y_task
# 原图看起来是单纯的下降趋势
y_collapse = 1.0 / (x + 0.2) 

ax1.plot(x, y_collapse, color='#E74C3C', linestyle='--', linewidth=3, label='Standard Loss')

# 去除上方和右方的边框 (Spines)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 设置标签
ax1.set_xlabel('Uncertainty Parameter $s$', fontsize=16)
ax1.set_ylabel('Loss Value', fontsize=16)
ax1.set_title('(a) Standard Uncertainty Weighting', y=-0.2, fontsize=16, fontweight='bold')

# 隐藏刻度值 (Ticks) - 原图没有数字
ax1.set_xticks([])
ax1.set_yticks([])

# 添加文本注释 (Annotations) - 字号已加大
ax1.text(0.2, 0.8 * max(y_collapse), 'Optimization Collapse', color='#E74C3C', fontsize=15, fontweight='bold')

# 添加箭头和文字: Lazy Learner
ax1.annotate('Lazy Learner\n($s \\to \infty$)', 
             xy=(2.4, y_collapse[-1]),  # 箭头指向末尾
             xytext=(1.5, y_collapse[-1] + 0.3), # 文字位置
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
             fontsize=14, ha='center')

# 图例
ax1.legend(frameon=False, loc='upper right', fontsize=12)


# ==========================================
# 右图：Prior-Regularized Uncertainty (Ours)
# ==========================================

# 1. 灰色点线: Weighted Task Loss
ax2.plot(x, y_task_weighted, color='#95A5A6', linestyle=':', linewidth=3, label='Weighted Task Loss')

# 2. 蓝色虚线: Prior Regularization
ax2.plot(x, y_prior, color='#3498DB', linestyle='--', linewidth=3, label='Prior Regularization')

# 3. 黑色实线: Total PR-UW Loss
ax2.plot(x, y_total, color='#2C3E50', linestyle='-', linewidth=3, label='Total PR-UW Loss')

# 4. 绿色圆点: Global Minimum
ax2.scatter(x_min, y_min, color='#2ECC71', s=100, zorder=5) # s是点的大小

# 去除边框
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 设置标签
ax2.set_xlabel('Uncertainty Parameter $s$', fontsize=16)
# ax2.set_ylabel('Loss Value', fontsize=16) # 右图通常不需要重复Y轴标签
ax2.set_title('(b) Prior-Regularized Uncertainty (Ours)', y=-0.2, fontsize=16, fontweight='bold')

# 隐藏刻度
ax2.set_xticks([])
ax2.set_yticks([])

# 添加文本注释
# Soft Anchor
ax2.text(1.6, 0.5 * max(y_total), 'Soft Anchor\n($\\beta s^2$)', color='#3498DB', fontsize=15, ha='center', fontweight='bold')

# Global Minimum (带箭头)
ax2.annotate('Global Minimum\n(Stable)', 
             xy=(x_min, y_min), 
             xytext=(x_min - 0.8, y_min + 0.3), 
             arrowprops=dict(facecolor='#2ECC71', edgecolor='#2ECC71', arrowstyle='->', lw=2),
             color='#2ECC71', fontsize=14, fontweight='bold', ha='center')

# 图例
ax2.legend(frameon=False, loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()

# 保存或显示
# plt.savefig('pr_loss_curve_large_font.png', dpi=300, bbox_inches='tight') # 如果你想保存图片，取消这行注释
plt.show()