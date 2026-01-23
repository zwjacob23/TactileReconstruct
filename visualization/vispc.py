import open3d as o3d
import numpy as np

def visualize_single_pcd(file_path, color_hex="#808080", point_size=3.0, estimate_normals=True):
    """
    高质量可视化单个点云（论文风格）。
    
    Args:
        file_path: 点云文件路径 (.pts, .xyz, .ply 等)
        color_hex: 十六进制颜色代码 (默认灰色 #808080，适合作为 Ground Truth)
                   如果你想画 Prediction，可以用蓝色 #0000FF
        point_size: 点的大小 (默认 3.0，根据物体疏密调整)
        estimate_normals: 是否计算法向量 (强烈建议 True，会有光照立体感)
    """
    print(f"[-] 正在加载: {file_path}")
    
    # 1. 鲁棒读取数据 (兼容 Open3D 和 Numpy 格式)
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0: raise Exception("Empty")
    except:
        # 如果读取失败，尝试作为纯文本 xyz 读取
        points = np.loadtxt(file_path)
        pcd = o3d.geometry.PointCloud()
        # 兼容 N*3 或 N*6 (带法向) 的数据
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    print(f"[-] 点云点数: {len(pcd.points)}")

    # 2. 设置统一颜色 (学术风)
    # 将 Hex 颜色转换为 RGB [0-1]
    color_hex = color_hex.lstrip('#')
    rgb = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    pcd.paint_uniform_color(rgb) 

    # 3. 计算法向量 (关键步骤！)
    # 如果点云本身没有法向量，计算它，这样渲染时会有光影效果，显得很立体
    if estimate_normals:
        print("[-] 正在计算法向量以增强渲染效果...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30)) # radius 需要根据你的物体尺度微调
        pcd.orient_normals_consistent_tangent_plane(100) # 让法向一致向外

    # 4. 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Paper Visualization', width=1024, height=768)
    
    vis.add_geometry(pcd)
    
    # 5. 渲染设置 (Render Options) - 让图更好看
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # 纯白背景 (论文标准)
    opt.point_size = point_size                  # 点的大小
    opt.light_on = True                          # 开启光照 (必须有法向量才生效)
    
    print("[-] 窗口已打开。")
    print("    [鼠标左键] 旋转")
    print("    [鼠标滚轮] 缩放")
    print("    [Ctrl + C] 复制当前视角截图到剪贴板 (Open3D自带功能)")
    print("    [Q] 退出")
    
    vis.run()
    vis.destroy_window()

# ================= 使用示例 =================
if __name__ == "__main__":
    # 替换你的 .pts 文件路径/good/014长方体.p
    my_file = "./example/薯片/0125薯片.pts" 
    #my_file = "./pc_label/乌龙.pts"
    
    # 模拟生成一个文件测试用 (如果你没有文件，直接运行会看到一个随机球体)
    # 实际使用时请注释掉下面这三行
    # dummy_points = np.random.rand(800, 3)
    # np.savetxt("test_dummy.pts", dummy_points)
    # my_file = "test_dummy.pts"

    # --- 场景 1: 画 Ground Truth (纯灰色/银色) ---
    visualize_single_pcd(my_file, color_hex="#A9A9A9", point_size=5.0)

    # --- 场景 2: 画 Prediction (如果你想单独展示预测结果，可以用蓝色) ---
    # visualize_single_pcd(my_file, color_hex="#0000FF", point_size=4.0)