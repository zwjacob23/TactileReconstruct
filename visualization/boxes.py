import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ================= 1. 数据准备 (保持不变) =================
objects_map_raw = [
    ('SportBtl',  0.04610),   # 海之言
    ('JuiceBtl',  0.04320),   # 水溶
    ('SodaBtl',   0.06840),   # 七喜
    ('Cuboid',    0.04630),   # 长方体
    ('WaterBtl',  0.05021),   # 怡宝
    ('WineGlass', 0.09000),   # 红酒杯
    ('Goblet',    0.07387),   # 高脚杯
    ('ChipTube',  0.05605),   # 薯片
    ('TeaBtl',    0.04730),   # 乌龙
    ('Cube',      0.08740),   # 正方体
    ('Cylinder',  0.07015),   # 圆柱体
    ('TriPrism',  0.16934),   # 三棱柱
]

# 数值乘以 100
labels = [item[0] for item in objects_map_raw]
cd_means = [item[1] * 100 for item in objects_map_raw] 
n_objects = len(labels)

# 模拟数据生成
np.random.seed(2024) 
data_cd = []
data_hd = []

for cd_mean in cd_means:
    if cd_mean > 10.0:
        sigma_cd = cd_mean * 0.2
    else:
        sigma_cd = cd_mean * 0.15
    sample_cd = np.random.normal(cd_mean, sigma_cd, 100)
    sample_cd = sample_cd[sample_cd > 0] 
    data_cd.append(sample_cd)

    hd_mean = cd_mean * np.random.uniform(3.5, 5.5)
    sigma_hd = hd_mean * 0.25 
    sample_hd = np.random.normal(hd_mean, sigma_hd, 100)
    sample_hd = sample_hd[sample_hd > 0]
    data_hd.append(sample_hd)

# ================= 2. 绘图设置 =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

c_cd_face = '#00796B'  
c_cd_edge = '#004D40'  
c_hd_face = '#E65100'  
c_hd_edge = '#BF360C'  

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# ================= 3. 绘图逻辑 =================
x_pos = np.arange(n_objects)
width = 0.25   
offset = 0.16  

# CD 箱体
bp1 = ax1.boxplot(data_cd, positions=x_pos - offset, widths=width, 
                  patch_artist=True, showfliers=True,
                  flierprops={'marker': 'o', 'markerfacecolor': c_cd_face, 'markeredgecolor':'none', 'alpha': 0.3, 'markersize': 3},
                  medianprops={'color': 'white', 'linewidth': 1.5},
                  boxprops={'facecolor': c_cd_face, 'edgecolor': c_cd_edge, 'linewidth': 1.2, 'alpha': 0.85},
                  whiskerprops={'color': c_cd_edge, 'linewidth': 1.2},
                  capprops={'color': c_cd_edge, 'linewidth': 1.2})

# HD 箱体
bp2 = ax2.boxplot(data_hd, positions=x_pos + offset, widths=width, 
                  patch_artist=True, showfliers=True,
                  flierprops={'marker': 'o', 'markerfacecolor': c_hd_face, 'markeredgecolor':'none', 'alpha': 0.3, 'markersize': 3},
                  medianprops={'color': 'white', 'linewidth': 1.5},
                  boxprops={'facecolor': c_hd_face, 'edgecolor': c_hd_edge, 'linewidth': 1.2, 'alpha': 0.85},
                  whiskerprops={'color': c_hd_edge, 'linewidth': 1.2},
                  capprops={'color': c_hd_edge, 'linewidth': 1.2})

# ================= 4. 坐标轴与图注调整 (核心修改) =================

ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.set_xlim(-0.6, n_objects - 0.4)

# Y轴标签
ax1.set_ylabel(r'Chamfer Distance (CD $\times 10^2$)', color=c_cd_edge, fontweight='bold', fontsize=14)
ax1.tick_params(axis='y', colors=c_cd_edge, labelsize=12)
ax1.set_ylim(bottom=0) 

ax2.set_ylabel(r'Hausdorff Distance (HD $\times 10^2$)', color=c_hd_edge, fontweight='bold', fontsize=14)
ax2.tick_params(axis='y', colors=c_hd_edge, labelsize=12)
ax2.set_ylim(bottom=0)

# 网格线
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
ax1.set_axisbelow(True)

# --- 图例设置 (修改部分) ---
legend_elements = [
    Patch(facecolor=c_cd_face, edgecolor=c_cd_edge, alpha=0.85, label=r'Chamfer Distance (CD)'),
    Patch(facecolor=c_hd_face, edgecolor=c_hd_edge, alpha=0.85, label=r'Hausdorff Distance (HD)')
]

# 将 loc 改为 'upper left'，并开启 frameon 加上背景框
ax1.legend(handles=legend_elements, 
           loc='upper left',           # 改为左上角
           bbox_to_anchor=(0.01, 0.99), # 微调位置：距离左边0.01，距离顶部0.99
           ncol=1,                     # 竖排 (设为2可以横排)
           frameon=True,               # 显示图例边框
           fancybox=False,             # 直角边框 (科研风格通常用直角)
           edgecolor='black',          # 边框颜色
           facecolor='white',          # 背景填充白色 (防止遮挡网格线变乱)
           framealpha=0.9,             # 轻微透明
           fontsize=11)

plt.tight_layout()
plt.savefig('scientific_dual_boxplot_inside_legend.pdf', dpi=300, bbox_inches='tight')
plt.savefig('scientific_dual_boxplot_inside_legend.png', dpi=300, bbox_inches='tight')
plt.show()