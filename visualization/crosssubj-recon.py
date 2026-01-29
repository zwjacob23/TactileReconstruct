import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 数据准备 =================
# 物体类别
categories = ['Goblet', 'JuiceBtl', 'Cube', 'ChipTube', 'SportBtl', 'Triprism','Average']

# CD Loss (数值单位: x 10^-2)
values_A = [0.84, 1.10, 1.00, 0.90, 0.98, 1.38, 0.97] # Within-Subject
values_B = [4.35, 5.13, 2.56, 1.52, 6.31, 6.99, 2.96] # Cross-Subject

# ================= 2. 全局绘图参数设置 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2 # 设置边框线宽

# 设置画布大小
fig, ax = plt.subplots(figsize=(10, 6))

# ================= 3. 绘制柱状图 =================
x = np.arange(len(categories))
width = 0.35

# Subject A (蓝色)
rects1 = ax.bar(
    x - width/2, values_A, width, 
    label='Baseline (Within-Subject)', 
    color='#2980b9', edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3
)

# Subject B (橙红色)
rects2 = ax.bar(
    x + width/2, values_B, width, 
    label='Cross-Subject(A->B)', 
    color='#d35400', edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3
)

# ================= 4. 细节美化与标注 =================

# --- 坐标轴标签 ---
ax.set_ylabel(r'Chamfer Distance ($\times 10^{-2}$)', fontsize=14, fontweight='bold', labelpad=10)

# --- X 轴设置 ---
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
# 全封闭风格下，保留刻度线（ticks）通常更好看，指向内部
ax.tick_params(axis='both', which='major', direction='in', length=5, width=1.2, top=True, right=True)

# --- 图例设置 ---
ax.legend(
    loc='upper left', fontsize=11, 
    frameon=True, framealpha=0.95, edgecolor='black', fancybox=False
)

# --- 网格线 ---
# 全封闭图表通常保留网格线，zorder=0 保证在柱子下面
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.5, zorder=0)

# --- 关键修改：四边全封闭设置 ---
# 显式开启顶部和右侧的刻度标签（如果不需要数字，只留刻度线，可以设 labeltop=False）
# 这里的代码已经通过 tick_params(top=True, right=True) 开启了四周的刻度线
# 边框本身因为没有执行 .set_visible(False) 操作，默认就是全封闭的

# --- 调整 Y 轴范围 ---
ax.set_ylim(0, max(values_B) * 1.15)

# ================= 5. 自动添加数值标签 =================
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')

autolabel(rects1)
autolabel(rects2)

# ================= 6. 保存与显示 =================
plt.tight_layout()
plt.savefig('cross_subject_barplot_box.pdf', bbox_inches='tight')
plt.savefig('cross_subject_barplot_box.png', dpi=300, bbox_inches='tight')
plt.show()