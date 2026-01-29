import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 生成假数据 (模拟不同频率的传感器信号)
# ==========================================
# 设置随机种子保证结果可复现
np.random.seed(42)

# 生成一个 0 到 40 秒的时间轴，分辨率高一些以便画出平滑曲线
t_total = np.linspace(0, 40, 2000)

# 定义三个不同频率的信号 (模拟 8Hz, 10Hz, 50Hz 的相对差异)
# 这里为了可视化清晰，没有用真实的 50Hz，而是用了相对的高中低频
# 添加一点点随机噪音让数据看起来更真实
freq1 = 0.5  # 低频模拟 (如手指压力)
signal1 = np.sin(2 * np.pi * freq1 * t_total) + 0.3 * np.sin(2 * np.pi * 1.2 * freq1 * t_total) + np.random.normal(0, 0.05, len(t_total))

freq2 = 1.2  # 中频模拟 (如手臂高度)
signal2 = np.sin(2 * np.pi * freq2 * t_total) + np.random.normal(0, 0.05, len(t_total))

freq3 = 3.0  # 高频模拟 (如关节角度)
signal3 = np.sin(2 * np.pi * freq3 * t_total) + 0.2 * np.cos(2 * np.pi * 0.5 * freq3 * t_total) + np.random.normal(0, 0.05, len(t_total))

# ==========================================
# 2. 定义可视化参数
# ==========================================
# 定义需要“保留”的有效时间窗口 (例如：从第 10 秒开始，持续 15 秒)
valid_start = 10
valid_duration = 15
valid_end = valid_start + valid_duration

# 定义同步点 (假设在有效窗口结束时对齐)
sync_point = valid_end

# 定义配色方案 (现代学术风)
colors = ['#1f77b4', '#2ca02c', '#d62728'] # 科技蓝, 清新绿, 珊瑚红
shade_color = 'gray'
shade_alpha = 0.3 # 半透明度

# 标签文本
labels = [
    "手指压力\n(Finger Pressure)",
    "手臂高度\n(Arm Height)",
    "关节角度\n(Joint Angle)"
]
freq_labels = ["8Hz", "10Hz", "50Hz"]

# ==========================================
# 3. 开始绘图
# ==========================================
# 创建 3行1列 的子图，共享 X 轴
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=False)

signals = [signal1, signal2, signal3]

# 遍历三个子图进行绘制
for i, ax in enumerate(axes):
    # --- 绘制波形 ---
    # 绘制整条波形，设置颜色和线宽
    ax.plot(t_total, signals[i], color=colors[i], linewidth=2, label="Valid Data" if i == 0 else "")
    
    # --- 绘制遮罩层 (核心步骤) ---
    # 使用 axvspan 绘制半透明矩形，覆盖无效区域
    # 区域 1: 开始到有效起点
    ax.axvspan(t_total[0], valid_start, color=shade_color, alpha=shade_alpha, label="Excluded" if i == 0 else "")
    # 区域 2: 有效终点到结束
    ax.axvspan(valid_end, t_total[-1], color=shade_color, alpha=shade_alpha)
    
    # --- 添加 "Excluded" 图标/文字 ---
    # 在左侧遮罩区添加一个“叉号框”示意
    bbox_props = dict(boxstyle="square,pad=0.3", fc="none", ec="black", lw=1, alpha=0.6)
    ax.text(valid_start/2, 0, "☒\nExcluded", ha="center", va="center", fontsize=10, color='black', alpha=0.7, bbox=bbox_props)
    # 在右侧遮罩区添加
    ax.text((valid_end + t_total[-1])/2, 0, "☒\nExcluded", ha="center", va="center", fontsize=10, color='black', alpha=0.7, bbox=bbox_props)

    # --- 添加标注和美化 ---
    # 设置 Y 轴标签
    ax.set_ylabel(labels[i], fontsize=12, fontname='Arial', labelpad=15)
    # 隐藏 Y 轴刻度 (让图看起来更干净，因为重点是趋势而非数值)
    ax.set_yticks([])
    # 添加网格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # 移除顶部和右侧的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加频率标注 (在右上方)
    ax.text(valid_end + 1, ax.get_ylim()[1]*0.8, freq_labels[i], fontsize=14, fontweight='bold', color='#333333')
    
    # --- 添加 "15s" 持续时间箭头 ---
    # 仅在最上面的图添加，避免冗余
    if i == 0:
        # 使用 annotate 画双向箭头
        ax.annotate('', xy=(valid_start, ax.get_ylim()[1]*1.1), xytext=(valid_end, ax.get_ylim()[1]*1.1),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        # 在箭头上方添加文字
        ax.text((valid_start + valid_end)/2, ax.get_ylim()[1]*1.25, f"{valid_duration}s", ha='center', fontsize=14)

# ==========================================
# 4. 添加全局元素 (同步线、图例等)
# ==========================================

# --- 绘制贯穿的同步虚线 ---
# 在所有子图上绘制一条垂直线
for ax in axes:
    ax.axvline(x=sync_point, color='black', linestyle='--', linewidth=2, alpha=0.8)

# 在最下面的图底部添加同步点说明
axes[-1].annotate('Synchronization Point', xy=(sync_point, axes[-1].get_ylim()[0]), xytext=(sync_point, axes[-1].get_ylim()[0]-0.5),
                 ha='center', va='top', fontsize=12, arrowprops=dict(arrowstyle='->', color='black'))

# --- 设置公共 X 轴标签 ---
axes[-1].set_xlabel("Time (s)", fontsize=14, fontname='Arial')
axes[-1].set_xlim(t_total[0], t_total[-1])

# --- 添加全局图例 ---
# 将图例放在最上方
fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=2, fontsize=12, frameon=True)

# 调整布局以防止标签重叠
plt.tight_layout()
# 留出顶部空间给图例和箭头
plt.subplots_adjust(top=0.9)

# 显示图像
plt.show()

# 如果要保存高清图片用于论文，取消下面这行的注释：
# plt.savefig("waveform_processing_pipeline.png", dpi=300, bbox_inches='tight')