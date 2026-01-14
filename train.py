import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
from contrastive import SupConLoss
# 引入第三方库 (确保安装了 chamferdist 或类似的库)
# 如果你使用的是自定义的 chamferdistance文件，请确保它在路径中
from chanmferdistance import ChamferDistance
from pcgrad import PCGrad
# 引入自定义模块
from config import get_args
from utils import save_point_cloud_as_pts, ensure_dirs
from loss import uncertainty_weighted_loss
from model import TactileTransformerMTL
from dataset import get_dataloaders
import random
import os
import numpy as np
import torch
# 在 train.py 顶部导入
from loss import uncertainty_weighted_loss
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

def setup_seed(seed=42):
    """
    固定所有可能的随机种子，确保实验可复现
    """
    # 1. 固定 Python 内置随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. 固定 NumPy 随机种子
    np.random.seed(seed)
    
    # 3. 固定 PyTorch CPU 和 GPU 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多 GPU
    
    # 4. 【关键】配置 CuDNN
    # deterministic=True 强制使用确定性卷积算法
    # benchmark=False 禁用自动寻找最快算法（因为不同算法在不同硬件上可能结果不同）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f">> [Info] Random seed set to {seed}. Results will be reproducible.")
# ... (之前的 import 保持不变) ...
# 引入上面的 calc_metrics_numpy 函数 (或者直接写在同一个文件里)
# from utils import calc_metrics_numpy 

# 假设你的 dataset.py 里有一个映射，或者你自己定义一个
# 这里的 ID 必须对应 target_shape 里的数值
CLASS_ID_TO_NAME = {
    0: "Cuboid",
    1: "Cube",
    2: "Cylinder",
    3: "TriPrism",
    4: "ChipTube",
    5: "WineGlass",
    6: "Goblet",
    7: "SportBtl",
    8: "TeaBtl",
    9: "SodaBtl",
    10: "JuiceBtl",
    11: "WaterBtl"
    # 根据你的实际数据集修改 ID 映射
}

def main():
    # ================= 1. 初始化设置 =================
    args = get_args()
    ensure_dirs(args)
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion_supcon = SupConLoss(temperature=0.07).to(device)
    print(f"Using device: {device}")

    # ================= 2. 准备数据 =================
    print("Loading data...")
    dataloader_train, dataloader_test = get_dataloaders(args)
    print(f"Data loaded. Train batches: {len(dataloader_train)}, Test batches: {len(dataloader_test)}")

    # ================= 3. 初始化模型与优化器 =================
    model = TactileTransformerMTL(args, use_contrastive=False).to(device)
    weight_contrastive = 0.1
    
    if args.PCGrad:
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = PCGrad(base_optimizer)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    chamfer_distance = ChamferDistance()
    criterion_radius = nn.MSELoss()
    criterion_theta = nn.MSELoss()

    # ================= 4. 定义全局监控变量 (按类别记录最佳) =================
    # 结构: { 'Cuboid': {'CD': inf, 'HD': inf, 'EMD': inf}, 'Cube': ... }
    best_class_metrics = {
        name: {'CD': float('inf'), 'HD': float('inf'), 'EMD': float('inf')} 
        for name in CLASS_ID_TO_NAME.values()
    }
    # 同时也记录一个全局平均的最佳
    best_avg_metrics = {'CD': float('inf'), 'HD': float('inf'), 'EMD': float('inf')}

    print('Start training...')
    
    # ================= 5. 主循环 =================
    for epoch in range(args.epoch):
        
        # ----------------- 训练阶段 (Training) -----------------
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader_train):
            inputs = [b.to(device) for b in batch[:3]]
            targets = [b.to(device) for b in batch[3:]]
            target_pc, target_shape, target_rad, target_theta = targets
            
            optimizer.zero_grad()
            outputs = model(*inputs)
            
            # Loss 计算
            loss_pc_raw = chamfer_distance(outputs['pointcloud'], target_pc)
            loss_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
            loss_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)

            if epoch < 50:
                loss_pc = loss_pc_raw * 20.0 
                loss_rad = loss_rad_raw * 1.0
                loss_theta = loss_theta_raw * 1.0
            else:
                loss_pc = uncertainty_weighted_loss(loss_pc_raw, model.log_var_pointcloud, regularization_weight=0.2)
                loss_rad = uncertainty_weighted_loss(loss_rad_raw, model.log_var_radius, regularization_weight=0.1)
                loss_theta = uncertainty_weighted_loss(loss_theta_raw, model.log_var_theta, regularization_weight=0.1)
            
            loss_contrastive = torch.tensor(0.0).to(device)
            # ... (对比学习逻辑保持不变) ...

            # Backprop
            total_loss = loss_pc + loss_rad + loss_theta + (weight_contrastive * loss_contrastive)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()

        print(f'Epoch [{epoch + 1}/{args.epoch}] Train Loss: {running_loss / len(dataloader_train):.4f}')

        # ----------------- 测试阶段 (每10个Epoch测一次) -----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            print(f"\n>>> 开始详细评估 (Epoch {epoch + 1}) ...")
            
            # 临时存储本 Epoch 的所有结果
            # 结构: { 'Cuboid': {'CD': [], 'HD': [], 'EMD': []}, ... }
            epoch_metrics_storage = {name: {'CD': [], 'HD': [], 'EMD': []} for name in CLASS_ID_TO_NAME.values()}

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_test):
                    inputs = [b.to(device) for b in batch[:3]]
                    targets = [b.to(device) for b in batch[3:]]
                    target_pc, target_shape, target_rad, target_theta = targets
                    
                    outputs = model(*inputs)
                    pred_pcs = outputs['pointcloud'] # [B, N, 3]

                    # === 逐样本计算指标 (为了准确分类和计算 EMD/HD) ===
                    batch_size = pred_pcs.shape[0]
                    for k in range(batch_size):
                        # 1. 获取当前样本的类别名称
                        cls_idx = target_shape[k].item()
                        cls_name = CLASS_ID_TO_NAME.get(cls_idx, "Unknown")
                        if cls_name == "Unknown": continue 

                        # 2. 准备数据 (转 CPU numpy)
                        pred_np = pred_pcs[k].cpu().numpy()
                        tgt_np = target_pc[k].cpu().numpy()

                        # 3. 计算 CD (用 GPU 版本，因为已经在 GPU 上了，这里取单样本 CD)
                        # 注意：chamfer_distance 通常返回 batch mean，这里我们手动算一下单样本的 sum/mean
                        # 或者为了方便，直接用上面那个 numpy 函数全家桶
                        # 这里为了统一标准，我们还是把 GPU 上的 CD 取下来
                        # 假设 chamfer_distance支持 batch计算，我们简单模拟单样本 loss
                        # 如果你的 chamfer_distance 不支持单样本，可以用 calc_metrics_numpy 里的占位符
                        
                        # 调用 Numpy 计算 HD 和 EMD
                        _, hd_val, emd_val = calc_metrics_numpy(pred_np, tgt_np)
                        
                        # 计算单样本 CD (复用 GPU 结果会快一点，但为了代码简单这里演示逻辑)
                        # 实际上 chamfer_distance(pred_pcs[k:k+1], target_pc[k:k+1])
                        cd_val = chamfer_distance(pred_pcs[k:k+1], target_pc[k:k+1]).item()

                        # 4. 存入临时列表
                        epoch_metrics_storage[cls_name]['CD'].append(cd_val)
                        epoch_metrics_storage[cls_name]['HD'].append(hd_val)
                        epoch_metrics_storage[cls_name]['EMD'].append(emd_val)

            # === 本 Epoch 结算 & 更新历史最佳 ===
            print(f"{'Category':<15} | {'CD':<10} | {'HD':<10} | {'EMD':<10}")
            print("-" * 55)
            
            total_avgs = {'CD': [], 'HD': [], 'EMD': []}

            for cls_name, metric_lists in epoch_metrics_storage.items():
                if len(metric_lists['CD']) == 0: continue

                # 计算该类别在本 Epoch 的平均值
                curr_cd = np.mean(metric_lists['CD'])
                curr_hd = np.mean(metric_lists['HD'])
                curr_emd = np.mean(metric_lists['EMD'])

                # 记录进总平均
                total_avgs['CD'].append(curr_cd)
                total_avgs['HD'].append(curr_hd)
                total_avgs['EMD'].append(curr_emd)

                # 更新历史最佳 (Lower is better)
                if curr_cd < best_class_metrics[cls_name]['CD']:
                    best_class_metrics[cls_name]['CD'] = curr_cd
                if curr_hd < best_class_metrics[cls_name]['HD']:
                    best_class_metrics[cls_name]['HD'] = curr_hd
                if curr_emd < best_class_metrics[cls_name]['EMD']:
                    best_class_metrics[cls_name]['EMD'] = curr_emd

                print(f"{cls_name:<15} | {curr_cd:.4f}     | {curr_hd:.4f}     | {curr_emd:.4f}")

            # 计算所有类别的平均 (Class-wise Average)
            avg_cd = np.mean(total_avgs['CD'])
            avg_hd = np.mean(total_avgs['HD'])
            avg_emd = np.mean(total_avgs['EMD'])
            
            # 更新全局最佳
            if avg_cd < best_avg_metrics['CD']: best_avg_metrics['CD'] = avg_cd
            if avg_hd < best_avg_metrics['HD']: best_avg_metrics['HD'] = avg_hd
            if avg_emd < best_avg_metrics['EMD']: best_avg_metrics['EMD'] = avg_emd

            print("-" * 55)
            print(f"{'AVERAGE':<15} | {avg_cd:.4f}     | {avg_hd:.4f}     | {avg_emd:.4f}")
            print("\n")

    # ================= 6. 训练结束总结 (最终大表) =================
    print("\n" + "="*30 + " FINAL BEST METRICS PER CATEGORY " + "="*30)
    print(f"{'Category':<15} | {'Best CD':<10} | {'Best HD':<10} | {'Best EMD':<10}")
    print("-" * 60)
    
    summary_lines = []
    header = f"{'Category':<15} | {'Best CD':<10} | {'Best HD':<10} | {'Best EMD':<10}"
    summary_lines.append(header)
    summary_lines.append("-" * 60)

    for cls_name in CLASS_ID_TO_NAME.values():
        vals = best_class_metrics.get(cls_name)
        if vals['CD'] == float('inf'): continue
        
        line = f"{cls_name:<15} | {vals['CD']:.4f}     | {vals['HD']:.4f}     | {vals['EMD']:.4f}"
        print(line)
        summary_lines.append(line)

    print("-" * 60)
    line_avg = f"{'AVERAGE':<15} | {best_avg_metrics['CD']:.4f}     | {best_avg_metrics['HD']:.4f}     | {best_avg_metrics['EMD']:.4f}"
    print(line_avg)
    summary_lines.append("-" * 60)
    summary_lines.append(line_avg)

    # 保存到 txt
    with open(os.path.join(args.test_predict_dir, "final_best_metrics.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print(f"[INFO] Final metrics saved to {args.test_predict_dir}")


if __name__ == "__main__":
    main()