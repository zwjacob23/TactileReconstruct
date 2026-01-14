import os

def save_point_cloud_as_pts(point_cloud, filename):
    """保存点云为 .pts 文件"""
    # 确保父目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as file:
        for point in point_cloud:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def calc_metrics_numpy(pred_np, gt_np):
    """
    输入:
        pred_np: (N, 3) numpy array
        gt_np:   (N, 3) numpy array
    输出:
        CD, HD, EMD (Scalar values)
    注意: EMD计算采用匈牙利算法(Hungarian Algorithm)匹配，计算量较大
    """
    # 1. Hausdorff Distance (双向)
    d1 = directed_hausdorff(pred_np, gt_np)[0]
    d2 = directed_hausdorff(gt_np, pred_np)[0]
    hd = max(d1, d2)

    # 2. Earth Mover's Distance (近似为最小匹配距离)
    # 计算距离矩阵
    dists = cdist(pred_np, gt_np)
    # 匈牙利算法找到最小权值匹配
    row_ind, col_ind = linear_sum_assignment(dists)
    emd = dists[row_ind, col_ind].mean()

    # 3. CD (为了验证与GPU版本的一致性，这里也可以算一下，或者直接用GPU的结果)
    # 这里为了速度省略，直接返回占位符，主循环用GPU算CD
    cd_cpu = 0.0 
    
    return cd_cpu, hd, emd
def ensure_dirs(args):
    """确保所有输出目录都存在"""
    os.makedirs(args.train_predict_dir, exist_ok=True)
    os.makedirs(args.train_label_dir, exist_ok=True)
    os.makedirs(args.test_predict_dir, exist_ok=True)
    os.makedirs(args.test_label_dir, exist_ok=True)