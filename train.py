import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys

# 引入模块
from contrastive import SupConLoss
# 回归使用标准的 Chamfer Distance，不用计算 HD 了，速度会更快
from chanmferdistance import ChamferDistance 
from pcgrad import PCGrad
from config import get_args
from utils import save_point_cloud_as_pts, ensure_dirs
from loss import uncertainty_weighted_loss
from model import TactileTransformerMTL
from dataset import get_dataloaders

# 引入 Refiner
from refinement import RefinementNetwork

def main():
    # ================= 1. 初始化设置 =================
    args = get_args()
    ensure_dirs(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ================= 2. 准备数据 =================
    print("Loading data...")
    dataloader_train, dataloader_test = get_dataloaders(args)
    print(f"Data loaded. Train batches: {len(dataloader_train)}, Test batches: {len(dataloader_test)}")

    # ================= 3. 初始化模型 (Stage 1 & Stage 2) =================
    # Stage 1: Transformer (Coarse + Uncertainty)
    model = TactileTransformerMTL(args, use_contrastive=True).to(device)
    
    # Stage 2: Refiner (Fine)
    refiner = RefinementNetwork(num_points=args.pointsNum).to(device)
    
    print(">> Initialized: Stage 1 (Transformer) + Stage 2 (Refiner)")

    # --- 优化器设置 ---
    all_parameters = list(model.parameters()) + list(refiner.parameters())
    
    if args.PCGrad:
        print(">> Using PCGrad Optimizer")
        base_optimizer = optim.Adam(all_parameters, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = PCGrad(base_optimizer)
    else:
        print(">> Using Standard Adam Optimizer")
        optimizer = optim.Adam(all_parameters, lr=args.lr, weight_decay=args.weight_decay)

    # ================= 4. 定义损失函数 =================
    criterion_supcon = SupConLoss(temperature=0.07).to(device)
    
    # 【修改】只使用标准的 Chamfer Distance
    criterion_pcd = ChamferDistance()
    
    criterion_radius = nn.MSELoss()
    criterion_theta = nn.MSELoss()
    
    # 权重
    weight_contrastive = 0.1

    # ================= 5. 定义全局监控变量 =================
    global_best_metric = float('inf')  
    global_best_info = {}

    print('Start training...')
    
    # ================= 6. 主循环 =================
    for epoch in range(args.epoch):
        
        # ----------------- 训练阶段 -----------------
        model.train()
        refiner.train() 
        
        running_loss = 0.0
        running_pc_coarse = 0.0
        running_pc_fine = 0.0
        
        for i, batch in enumerate(dataloader_train):
            inputs = [b.to(device) for b in batch[:3]]
            targets = [b.to(device) for b in batch[3:]]
            target_pc, target_shape, target_rad, target_theta = targets
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # 1. Stage 1 输出
            outputs = model(*inputs)
            coarse_pc = outputs['pointcloud']
            # 获取 Uncertainty Map [B, N, 1]
            unc_map = outputs['uncertainty_map'] 
            global_feat = outputs['shared_features']
            # 2. Stage 2 精修 (输入: 坐标 + 地图)
            fine_pc = refiner(coarse_pc.detach(), unc_map, global_feat)
            
            # --- Loss Calculation ---
            
            # (A) Coarse Loss (Stage 1 监督)
            l_coarse_raw = criterion_pcd(coarse_pc, target_pc)
            l_coarse = uncertainty_weighted_loss(l_coarse_raw, model.log_var_pointcloud)
            l_unc_reg = torch.mean(unc_map) * 0.001
            # (B) Fine Loss (Stage 2 监督)
            # 这里是最终输出，我们希望它越准越好
            l_fine_raw = criterion_pcd(fine_pc, target_pc)
            # 这里可以选择是否加权，简单起见，作为硬约束不加权，或者共用 log_var
            # 既然是最终目标，直接作为主 Loss
            l_fine = l_fine_raw + l_unc_reg
            
            # (C) 辅助任务 Loss
            l_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
            l_rad = uncertainty_weighted_loss(l_rad_raw, model.log_var_radius)
            
            l_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)
            l_theta = uncertainty_weighted_loss(l_theta_raw, model.log_var_theta)
            
            # (D) 对比 Loss
            l_contrastive = torch.tensor(0.0).to(device)
            if 'contrastive_embed' in outputs:
                 features = outputs['contrastive_embed']
                 if features.shape[0] > 1:
                     l_contrastive = criterion_supcon(features, target_shape)

            # --- Backward Pass ---
            if args.PCGrad:
                # 任务列表: [Coarse, Fine, Rad, Theta, Contrastive]
                losses = [l_coarse, l_fine, l_rad, l_theta]
                
                if l_contrastive.item() > 0:
                    losses.append(l_contrastive * weight_contrastive)
                
                optimizer.pc_backward(losses)
                optimizer.step()
                total_loss = sum(losses)
            else:
                total_loss = l_coarse + l_fine + l_rad + l_theta + \
                             (weight_contrastive * l_contrastive)
                total_loss.backward()
                optimizer.step()
            
            # 记录
            running_loss += total_loss.item()
            running_pc_coarse += l_coarse_raw.item()
            running_pc_fine += l_fine_raw.item()

        # 打印训练日志
        avg_train_loss = running_loss / len(dataloader_train)
        w_pc = torch.exp(-model.log_var_pointcloud).item()
        
        print(f'Epoch [{epoch + 1}/{args.epoch}] Train Loss: {avg_train_loss:.4f} '
              f'| Coarse: {running_pc_coarse/len(dataloader_train):.4f} (w={w_pc:.2f}) '
              f'| Fine: {running_pc_fine/len(dataloader_train):.4f}')

        # ----------------- 测试阶段 -----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            refiner.eval()
            print(f"\n>>> 开始测试 (Epoch {epoch + 1}) ...")
            
            test_fine_sum = 0.0
            test_coarse_sum = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_test):
                    inputs = [b.to(device) for b in batch[:3]]
                    targets = [b.to(device) for b in batch[3:]]
                    target_pc, _, _, _ = targets
                    
                    # Forward
                    outputs = model(*inputs)
                    coarse_pc = outputs['pointcloud']
                    unc_map = outputs['uncertainty_map']
                    global_feat = outputs['shared_features']
            # 2. Stage 2 精修 (输入: 坐标 + 地图)
                    fine_pc = refiner(coarse_pc, unc_map, global_feat)
                    
                    # Metrics (Standard Chamfer)
                    l_c_raw = criterion_pcd(coarse_pc, target_pc)
                    l_f_raw = criterion_pcd(fine_pc, target_pc)
                    
                    test_coarse_sum += l_c_raw.item()
                    test_fine_sum += l_f_raw.item()
                    
                    # Best Batch Logic (基于 Fine Loss)
                    current_metric = l_f_raw.item()
                    
                    if current_metric < global_best_metric:
                        prev = global_best_metric
                        global_best_metric = current_metric
                        
                        # 记录所有关键信息
                        global_best_info = {
                            'epoch': epoch + 1,
                            'batch_idx': batch_idx,
                            'best_metric_fine': global_best_metric,
                            'metric_coarse': l_c_raw.item()
                        }
                        
                        print(f"  [New Best!] Batch {batch_idx}: Fine Loss {prev:.4f} -> {global_best_metric:.4f} (Coarse: {l_c_raw.item():.4f})")
                        
                        # 保存
                        for k in range(min(5, inputs[0].shape[0])):
                            save_point_cloud_as_pts(fine_pc[k].cpu().numpy(), os.path.join(args.test_predict_dir, f'best_sample_{k}.pts'))
                            save_point_cloud_as_pts(target_pc[k].cpu().numpy(), os.path.join(args.test_label_dir, f'best_sample_{k}.pts'))
                            
            print(f">>> Epoch {epoch + 1} Summary: Avg Coarse: {test_coarse_sum/len(dataloader_test):.4f} | Avg Fine: {test_fine_sum/len(dataloader_test):.4f}")
            print("-" * 60 + "\n")

    # ================= 6. 总结 =================
    print("\n" + "="*30 + " TRAINING FINISHED " + "="*30)
    if global_best_info:
        print(f"全过程最佳记录 (Global Best Record):")
        print(f"  - Epoch: {global_best_info['epoch']}")
        print(f"  - Batch Index: {global_best_info['batch_idx']}")
        print(f"  - Best Fine Loss: {global_best_info['best_metric_fine']:.6f}")
        print(f"  - (Original Coarse Loss: {global_best_info['metric_coarse']:.6f})")
        print(f"最佳结果文件已保存在: {args.test_predict_dir}")
    else:
        print("未记录到任何有效测试数据。")
    print("="*80)

if __name__ == "__main__":
    main()