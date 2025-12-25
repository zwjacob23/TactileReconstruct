import os

def save_point_cloud_as_pts(point_cloud, filename):
    """保存点云为 .pts 文件"""
    # 确保父目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as file:
        for point in point_cloud:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

def ensure_dirs(args):
    """确保所有输出目录都存在"""
    os.makedirs(args.train_predict_dir, exist_ok=True)
    os.makedirs(args.train_label_dir, exist_ok=True)
    os.makedirs(args.test_predict_dir, exist_ok=True)
    os.makedirs(args.test_label_dir, exist_ok=True)