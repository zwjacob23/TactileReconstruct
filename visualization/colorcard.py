import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 设置绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
})

# 创建一个 Figure，专门用于放置色卡
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# 定义 colormap，使用 'jet' 看起来与原图比较接近 (蓝->青->黄->红)
cmap = mpl.cm.jet
# 定义误差范围，这里假设一个合理的值，比如 0 到 0.05 米
norm = mpl.colors.Normalize(vmin=0.0, vmax=0.05)

# 创建 Colorbar
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, orientation='horizontal')
cb.set_label('Reconstruction Error (m)', fontweight='bold')

# 设置刻度
cb.set_ticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
cb.set_ticklabels(['0.00', '0.01', '0.02', '0.03', '0.04', '≥0.05'])

# 保存并显示
plt.savefig('colorbar.png', dpi=300, bbox_inches='tight')
plt.show()