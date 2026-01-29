import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def draw_error_heatmap(pred_path, gt_path, max_dist_threshold=0.05):
    """
    绘制点云误差热力图。
    
    Args:
        pred_path: 预测点云 (.pts 或 .xyz) 的路径。
        gt_path: Ground Truth 点云 (.pts 或 .xyz) 的路径。
        max_dist_threshold: 误差截断阈值。大于这个距离的误差都显示为最红色。
                            这个值需要根据你的物体实际尺度来调整。
                            比如物体大小是1，这个值可能设为0.02~0.05比较合适。
    """
    # --- 1. 读取数据 ---
    print(f"正在加载预测点云: {pred_path}")
    # 尝试用 Open3D 直接读取，如果不行则用 numpy 读取文本再转换
    try:
        pcd_pred = o3d.io.read_point_cloud(pred_path)
        # 如果读取后点数为0，说明格式可能不对，尝试用numpy读取通用格式
        if len(pcd_pred.points) == 0:
            raise Exception("Open3D读取失败，尝试Numpy方式")
    except:
        points_pred = np.loadtxt(pred_path)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points_pred[:, :3])

    print(f"正在加载 GT 点云: {gt_path}")
    try:
        pcd_gt = o3d.io.read_point_cloud(gt_path)
        if len(pcd_gt.points) == 0:
            raise Exception("Open3D读取失败，尝试Numpy方式")
    except:
        points_gt = np.loadtxt(gt_path)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt[:, :3])

    print(f"预测点数: {len(pcd_pred.points)}, GT点数: {len(pcd_gt.points)}")

    # --- 2. 计算核心误差 (最近邻距离) ---
    print("正在计算点到点的最近距离...")
    # compute_point_cloud_distance 计算源点云中每个点到目标点云最近点的距离
    dists = pcd_pred.compute_point_cloud_distance(pcd_gt)
    dists = np.asarray(dists)
    
    print(f"最小误差: {np.min(dists):.4f}, 最大误差: {np.max(dists):.4f}, 平均误差: {np.mean(dists):.4f}")

    # --- 3. 颜色映射 (Color Mapping) ---
    # 重要：归一化距离。将距离映射到 0-1 之间。
    # 我们使用一个阈值来截断，防止个别离谱的噪点导致整体颜色对比度降低。
    # 任何大于 max_dist_threshold 的误差都会被视为最大误差 (1.0)
    normalized_dists = np.clip(dists / max_dist_threshold, 0, 1)

    # 使用 matplotlib 的 colormap。'jet' 是经典的蓝-青-黄-红热力图。
    # 你也可以换成 'plasma', 'viridis', 'coolwarm' 等。
    colormap = plt.get_cmap('jet')
    # colormap 接收 0-1 的输入，输出 RGBA (N, 4)
    colors_rgba = colormap(normalized_dists)
    # Open3D 只需要 RGB，去掉 Alpha 通道
    colors_rgb = colors_rgba[:, :3]

    # --- 4. 渲染展示 ---
    # 将计算好的颜色赋值给预测点云
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_rgb)

    print(f"绘图完毕。截断阈值设置为: {max_dist_threshold}")
    print("弹窗中，蓝色=误差小，红色=误差大。请使用鼠标旋转查看。")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Error Heatmap', width=800, height=600)
    vis.add_geometry(pcd_pred)
    
    # 为了好看，设置背景为白色，点稍微大一点
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # 白色背景
    opt.point_size = 5 # 设置点的大小
    
    vis.run()
    vis.destroy_window()

# =========================================
# 使用示例
# =========================================
# 请替换成你实际的文件路径
# 假设你有一个瓶子的预测和真值
pred_file = "./example/crossSubject/insub高脚杯.pts" 
gt_file = "./pc_label/高脚杯.pts"

# 创建一些假数据用于演示 (如果你没有现成的文件，取消下面注释来测试代码)
# def create_dummy_data(filename, noise_level):
#     points = np.random.rand(1000, 3) # 生成1000个随机点
#     if noise_level > 0:
#         points += np.random.normal(0, noise_level, points.shape)
#     np.savetxt(filename, points, fmt='%.6f')
# create_dummy_data(gt_file, 0)          # 生成完美的GT
# create_dummy_data(pred_file, 0.02)     # 生成带有噪声的预测

# 运行绘图函数
# 重要：根据你的物体尺寸调整 max_dist_threshold
# 如果你的物体坐标是在 0-1 之间，0.05 意味着 5% 的误差就显示为最红
draw_error_heatmap(pred_file, gt_file, max_dist_threshold=0.015)