import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import numpy as np
import random
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.optimize import linear_sum_assignment

# 你的自定义库
from contrastive import SupConLoss
from chanmferdistance import ChamferDistance
from pcgrad import PCGrad
from config import get_args
from utils import save_point_cloud_as_pts, ensure_dirs
from model import TactileTransformerMTL
from dataset import get_dataloaders
from loss import uncertainty_weighted_loss

# ==========================================
# 1. ID 映射 (必须与 dataset.py 的列表顺序一致)
# ==========================================
CLASS_ID_TO_NAME = {
    0: 'Cuboid',     # 长方体
    1: 'Cube',       # 正方体
    2: 'Goblet',     # 高脚杯
    3: 'WineGlass',  # 红酒杯
    4: 'TriPrism',   # 三棱柱
    5: 'Cylinder',   # 圆柱体
    6: 'ChipTube',   # 薯片
    7: 'SportBtl',   # 海之言
    8: 'TeaBtl',     # 乌龙
    9: 'SodaBtl',    # 七喜
    10: 'JuiceBtl',  # 水溶
    11: 'WaterBtl'   # 怡宝
}

# ==========================================
# 2. 辅助函数：计算 HD 和 EMD (Numpy CPU版)
# ==========================================
def calc_metrics_numpy(pred_np, gt_np):
    """
    计算 Hausdorff Distance (HD) 和 Earth Mover's Distance (EMD)
    注意：EMD 计算较慢，适合评估时使用
    """
    # 1. Hausdorff Distance
    d1 = directed_hausdorff(pred_np, gt_np)[0]
    d2 = directed_hausdorff(gt_np, pred_np)[0]
    hd = max(d1, d2)

    # 2. EMD (近似匹配)
    dists = cdist(pred_np, gt_np)
    row_ind, col_ind = linear_sum_assignment(dists)
    emd = dists[row_ind, col_ind].mean()
    
    return hd, emd

def setup_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> [Info] Random seed set to {seed}.")

def main():
    args = get_args()
    ensure_dirs(args)
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    dataloader_train, dataloader_test = get_dataloaders(args)
    print(f"Data loaded. Train batches: {len(dataloader_train)}, Test batches: {len(dataloader_test)}")

    # 模型初始化
    model = TactileTransformerMTL(args, use_contrastive=False).to(device)
    weight_contrastive = 0.1
    
    if args.PCGrad:
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = PCGrad(base_optimizer)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss 定义
    chamfer_distance = ChamferDistance()
    criterion_radius = nn.MSELoss()
    criterion_theta = nn.MSELoss()
    criterion_supcon = SupConLoss(temperature=0.07).to(device)

    # 记录每个类别历史最好的指标
    best_class_metrics = {
        name: {'CD': float('inf'), 'HD': float('inf'), 'EMD': float('inf')} 
        for name in CLASS_ID_TO_NAME.values()
    }
    # 记录全局平均最好
    best_avg_metrics = {'CD': float('inf'), 'HD': float('inf'), 'EMD': float('inf')}

    print('Start training...')
    
    for epoch in range(args.epoch):
        # ----------------- Training -----------------
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader_train):
            # 解包：注意现在 target_obj_idx 是第 8 个元素，但训练用不到它
            inputs = [b.to(device) for b in batch[:3]]
            targets = [b.to(device) for b in batch[3:]]
            target_pc, target_shape, target_rad, target_theta, _ = targets # 忽略 obj_idx
            
            optimizer.zero_grad()
            outputs = model(*inputs)
            
            # Loss 计算
            loss_pc_raw = chamfer_distance(outputs['pointcloud'], target_pc)
            loss_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
            loss_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)

            # 动态权重策略
            if epoch < 50:
                loss_pc = loss_pc_raw * 20.0 
                loss_rad = loss_rad_raw * 1.0
                loss_theta = loss_theta_raw * 1.0
            else:
                loss_pc = uncertainty_weighted_loss(loss_pc_raw, model.log_var_pointcloud, regularization_weight=0.2)
                loss_rad = uncertainty_weighted_loss(loss_rad_raw, model.log_var_radius, regularization_weight=0.1)
                loss_theta = uncertainty_weighted_loss(loss_theta_raw, model.log_var_theta, regularization_weight=0.1)
            
            # 对比学习
            loss_contrastive = torch.tensor(0.0).to(device)
            if 'contrastive_embed' in outputs:
                features = outputs['contrastive_embed']
                labels = target_shape
                if features.shape[0] > 1:
                    loss_contrastive = criterion_supcon(features, labels)

            # 反向传播
            if args.PCGrad:
                losses = [loss_pc, loss_rad, loss_theta]
                if loss_contrastive.item() > 0:
                    losses.append(loss_contrastive * weight_contrastive)
                optimizer.pc_backward(losses)
                optimizer.step()
            else:
                total_loss = loss_pc + loss_rad + loss_theta + (weight_contrastive * loss_contrastive)
                total_loss.backward()
                optimizer.step()
            
            running_loss += (loss_pc + loss_rad + loss_theta).item()

        print(f'Epoch [{epoch + 1}/{args.epoch}] Train Loss: {running_loss / len(dataloader_train):.4f}')

        # ----------------- Testing (每10轮) -----------------
        if (epoch + 1) % 100 == 0:
            model.eval()
            print(f"\n>>> 开始详细评估 (Epoch {epoch + 1}) ...")
            
            # 临时存储本 Epoch 的数据
            epoch_metrics_storage = {name: {'CD': [], 'HD': [], 'EMD': []} for name in CLASS_ID_TO_NAME.values()}

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_test):
                    inputs = [b.to(device) for b in batch[:3]]
                    targets = [b.to(device) for b in batch[3:]]
                    # 这里接住 target_obj_idx
                    target_pc, target_shape, target_rad, target_theta, target_obj_idx = targets
                    
                    outputs = model(*inputs)
                    pred_pcs = outputs['pointcloud']

                    # === 逐样本计算指标 ===
                    batch_size = pred_pcs.shape[0]
                    for k in range(batch_size):
                        # 获取唯一物体ID (0-11)
                        obj_unique_id = target_obj_idx[k].item()
                        cls_name = CLASS_ID_TO_NAME.get(obj_unique_id, "Unknown")
                        
                        if cls_name == "Unknown": continue 

                        # 准备数据
                        pred_np = pred_pcs[k].cpu().numpy()
                        tgt_np = target_pc[k].cpu().numpy()

                        # 1. 计算 CD (使用GPU结果)
                        cd_val = chamfer_distance(pred_pcs[k:k+1], target_pc[k:k+1]).item()
                        
                        # 2. 计算 HD 和 EMD (CPU)
                        hd_val, emd_val = calc_metrics_numpy(pred_np, tgt_np)

                        # 存入列表
                        epoch_metrics_storage[cls_name]['CD'].append(cd_val)
                        epoch_metrics_storage[cls_name]['HD'].append(hd_val)
                        epoch_metrics_storage[cls_name]['EMD'].append(emd_val)

            # === 本 Epoch 结算 ===
            print(f"{'Category':<15} | {'CD':<10} | {'HD':<10} | {'EMD':<10}")
            print("-" * 55)
            
            total_avgs = {'CD': [], 'HD': [], 'EMD': []}

            for cls_id in sorted(CLASS_ID_TO_NAME.keys()): # 按顺序遍历
                cls_name = CLASS_ID_TO_NAME[cls_id]
                metrics = epoch_metrics_storage[cls_name]
                
                if len(metrics['CD']) == 0: 
                    continue

                curr_cd = np.mean(metrics['CD'])
                curr_hd = np.mean(metrics['HD'])
                curr_emd = np.mean(metrics['EMD'])

                total_avgs['CD'].append(curr_cd)
                total_avgs['HD'].append(curr_hd)
                total_avgs['EMD'].append(curr_emd)

                # 更新历史最佳
                if curr_cd < best_class_metrics[cls_name]['CD']: best_class_metrics[cls_name]['CD'] = curr_cd
                if curr_hd < best_class_metrics[cls_name]['HD']: best_class_metrics[cls_name]['HD'] = curr_hd
                if curr_emd < best_class_metrics[cls_name]['EMD']: best_class_metrics[cls_name]['EMD'] = curr_emd

                print(f"{cls_name:<15} | {curr_cd:.4f}     | {curr_hd:.4f}     | {curr_emd:.4f}")

            # 计算全局平均
            if len(total_avgs['CD']) > 0:
                avg_cd = np.mean(total_avgs['CD'])
                avg_hd = np.mean(total_avgs['HD'])
                avg_emd = np.mean(total_avgs['EMD'])
                
                if avg_cd < best_avg_metrics['CD']: best_avg_metrics['CD'] = avg_cd
                if avg_hd < best_avg_metrics['HD']: best_avg_metrics['HD'] = avg_hd
                if avg_emd < best_avg_metrics['EMD']: best_avg_metrics['EMD'] = avg_emd

                print("-" * 55)
                print(f"{'AVERAGE':<15} | {avg_cd:.4f}     | {avg_hd:.4f}     | {avg_emd:.4f}")
            print("\n")

    # ================= 6. 训练结束总结 =================
    print("\n" + "="*30 + " FINAL BEST METRICS " + "="*30)
    
    # 定义表头
    header = f"{'Category':<15} | {'Best CD':<12} | {'Best HD':<12} | {'Best EMD':<12}"
    print(header)
    print("-" * 65)
    
    # 准备写入文件的内容列表
    summary_lines = []
    summary_lines.append(f"Training Finished. Total Epochs: {args.epoch}")
    summary_lines.append("=" * 65)
    summary_lines.append(header)
    summary_lines.append("-" * 65)

    # 遍历所有 12 个物体，按顺序打印和记录
    for cls_id in sorted(CLASS_ID_TO_NAME.keys()):
        cls_name = CLASS_ID_TO_NAME[cls_id]
        vals = best_class_metrics[cls_name]
        
        # 如果某个物体一次都没测到（比如数据集里没有），就跳过
        if vals['CD'] == float('inf'): 
            continue
        
        # 格式化每一行：保留4位小数
        line = f"{cls_name:<15} | {vals['CD']:.4f}       | {vals['HD']:.4f}       | {vals['EMD']:.4f}"
        
        # 打印到控制台
        print(line)
        # 添加到文件列表
        summary_lines.append(line)

    # 添加分隔线和平均值
    print("-" * 65)
    summary_lines.append("-" * 65)
    
    # 记录全局平均值 (Average of Best)
    # 注意：这里计算的是“每个物体历史最佳值的平均”，代表了模型的上限潜力
    avg_line = f"{'AVERAGE':<15} | {best_avg_metrics['CD']:.4f}       | {best_avg_metrics['HD']:.4f}       | {best_avg_metrics['EMD']:.4f}"
    print(avg_line)
    summary_lines.append(avg_line)
    
    # --- 核心保存步骤 ---
    save_path = os.path.join(args.test_predict_dir, f"final_best_metrics_for{args.nmode}.txt")
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        
    print(f"\n[SUCCESS] 最终结果已保存至: {save_path}")
    print("="*80)

if __name__ == "__main__":
    main()