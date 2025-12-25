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

def main():
    # ================= 1. 初始化设置 =================
    args = get_args()
    ensure_dirs(args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion_supcon = SupConLoss(temperature=0.07).to(device)
    print(f"Using device: {device}")

    # ================= 2. 准备数据 =================
    print("Loading data...")
    dataloader_train, dataloader_test = get_dataloaders(args)
    print(f"Data loaded. Train batches: {len(dataloader_train)}, Test batches: {len(dataloader_test)}")

    # ================= 3. 初始化模型与优化器 =================
    model = TactileTransformerMTL(args, use_contrastive=True).to(device)
    weight_contrastive = 0.1 ## 对比学习超参 对比学习的权重
    if args.PCGrad: # 如果打开了PCGrad
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = PCGrad(base_optimizer)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 定义损失函数
    chamfer_distance = ChamferDistance()
    criterion_radius = nn.MSELoss()
    criterion_theta = nn.MSELoss()
    # criterion_shape = nn.CrossEntropyLoss() # 已禁用

    # ================= 4. 定义全局监控变量 =================
    # 用于记录整个训练过程中，表现最好的那个Batch的详细信息
    global_best_batch_loss = float('inf')  
    global_best_info = {
        'epoch': -1,
        'batch_idx': -1,
        'total_loss': float('inf'),
        'loss_pc_weighted': 0.0,
        'loss_rad_weighted': 0.0,
        'loss_theta_weighted': 0.0,
        'loss_pc_raw': 0.0,
        'loss_rad_raw': 0.0,
        'loss_theta_raw': 0.0
    }

    print('Start training...')
    
    # ================= 5. 主循环 =================
    for epoch in range(args.epoch):
        
        # ----------------- 训练阶段 (Training) -----------------
        model.train()
        running_loss = 0.0
        running_pc_raw = 0.0
        running_rad_raw = 0.0
        running_theta_raw = 0.0
        
        for i, batch in enumerate(dataloader_train):
            # 解包数据 (前3个是输入，后4个是标签)
            inputs = [b.to(device) for b in batch[:3]]
            # targets: [pointcloud, shape, radius_seq, theta_seq]
            targets = [b.to(device) for b in batch[3:]]
            target_pc, target_shape, target_rad, target_theta = targets
            
            optimizer.zero_grad()
            outputs = model(*inputs)
            
            # 1. 计算原始 Loss
            loss_pc_raw = chamfer_distance(outputs['pointcloud'], target_pc)
            loss_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
            loss_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)
            
            # 2. 计算 Uncertainty Weighted Loss
            loss_pc = uncertainty_weighted_loss(loss_pc_raw, model.log_var_pointcloud)
            loss_rad = uncertainty_weighted_loss(loss_rad_raw, model.log_var_radius)
            loss_theta = uncertainty_weighted_loss(loss_theta_raw, model.log_var_theta)
            
            # 3. 计算对比学习loss
            loss_contrastive = torch.tensor(0.0).to(device)
            if 'contrastive_embed' in outputs:
                features = outputs['contrastive_embed'] # [Batch, 128]
                labels = target_shape                   # [Batch]
            
                # 只有当 Batch 里至少有两个样本时才能算对比
                if features.shape[0] > 1:
                    loss_contrastive = criterion_supcon(features, labels)
            
            if args.PCGrad:
                
                losses = [loss_pc, loss_rad, loss_theta]
                if loss_contrastive.item() > 0 and args.PCGrad:
                    losses.append(loss_contrastive * weight_contrastive)
                total_loss = sum(losses)
                optimizer.zero_grad()
                optimizer.pc_backward(losses) # <--- 使用 PCGrad 的 backward
                optimizer.step()
            else:
                total_loss = loss_pc + loss_rad + loss_theta + (weight_contrastive * loss_contrastive)
                total_loss.backward()
                optimizer.step()
            
            # 记录数据
            running_loss += total_loss.item()
            running_pc_raw += loss_pc_raw.item()
            running_rad_raw += loss_rad_raw.item()
            running_theta_raw += loss_theta_raw.item()

        # 计算训练集 Epoch 平均值
        avg_train_loss = running_loss / len(dataloader_train)
        
        # 获取当前的不确定性权重用于打印
        with torch.no_grad():
            w_pc = torch.exp(-model.log_var_pointcloud).item()
            w_ra = torch.exp(-model.log_var_radius).item()
            w_th = torch.exp(-model.log_var_theta).item()

        print(f'Epoch [{epoch + 1}/{args.epoch}] Train Loss: {avg_train_loss:.4f} '
              f'| PC_Raw: {running_pc_raw/len(dataloader_train):.4f} (w={w_pc:.2f}) '
              f'| Rad_Raw: {running_rad_raw/len(dataloader_train):.4f} (w={w_ra:.2f}) '
              f'| Theta_Raw: {running_theta_raw/len(dataloader_train):.4f} (w={w_th:.2f})')

        # ----------------- 测试阶段 (每10个Epoch测一次) -----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            print(f"\n>>> 开始测试 (Epoch {epoch + 1}) ...")
            
            test_epoch_loss = 0.0
            test_pc_raw_sum = 0.0
            test_rad_raw_sum = 0.0
            test_theta_raw_sum = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_test):
                    inputs = [b.to(device) for b in batch[:3]]
                    targets = [b.to(device) for b in batch[3:]]
                    target_pc, _, target_rad, target_theta = targets
                    
                    outputs = model(*inputs)
                    
                    # 1. 计算原始 Loss
                    l_pc_raw = chamfer_distance(outputs['pointcloud'], target_pc)
                    l_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
                    l_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)
                    
                    # 2. 计算加权 Loss
                    l_pc = uncertainty_weighted_loss(l_pc_raw, model.log_var_pointcloud)
                    l_rad = uncertainty_weighted_loss(l_rad_raw, model.log_var_radius)
                    l_th = uncertainty_weighted_loss(l_theta_raw, model.log_var_theta)
                    
                    current_batch_total_loss = l_pc_raw
                    
                    # 记录 Epoch 统计
                    test_epoch_loss += current_batch_total_loss.item()
                    test_pc_raw_sum += l_pc_raw.item()
                    test_rad_raw_sum += l_rad_raw.item()
                    test_theta_raw_sum += l_theta_raw.item()
                    
                    # =============== 核心：判断是否打破历史最佳Batch记录 ===============
                    if current_batch_total_loss.item() < global_best_batch_loss:
                        prev_best = global_best_batch_loss
                        global_best_batch_loss = current_batch_total_loss.item()
                        
                        # 更新全局信息字典
                        global_best_info = {
                            'epoch': epoch + 1,
                            'batch_idx': batch_idx,
                            'total_loss': global_best_batch_loss,
                            'loss_pc_weighted': l_pc.item(),
                            'loss_rad_weighted': l_rad.item(),
                            'loss_theta_weighted': l_th.item(),
                            'loss_pc_raw': l_pc_raw.item(),
                            'loss_rad_raw': l_rad_raw.item(),
                            'loss_theta_raw': l_theta_raw.item()
                        }
                        
                        # 打印详细信息
                        print(f"  [New Best Batch!] Epoch {epoch+1} | Batch {batch_idx}")
                        print(f"  Loss Dropped: {prev_best:.6f} -> {global_best_batch_loss:.6f}")
                       # 加上 .item() 提取数值
                        print(f"  Details (Weighted): PC={l_pc.item():.4f}, Rad={l_rad.item():.4f}, Theta={l_th.item():.4f}")
                        
                        # 保存当前 Batch 的点云 (取前5个样本)
                        for k in range(min(5, inputs[0].shape[0])):
                            pc_np = outputs['pointcloud'][k].cpu().numpy()
                            tgt_np = target_pc[k].cpu().numpy()
                            
                            # 文件名: best_sample_{k}.pts (始终覆盖旧的 best，保证文件夹里只有最好的)
                            # 如果你想保留历史记录，可以在文件名里加上 epoch: f'best_ep{epoch+1}_sample_{k}.pts'
                            pred_path = os.path.join(args.test_predict_dir, f'best_sample_{k}.pts')
                            label_path = os.path.join(args.test_label_dir, f'best_sample_{k}.pts')
                            
                            save_point_cloud_as_pts(pc_np, pred_path)
                            save_point_cloud_as_pts(tgt_np, label_path)
                            
                        print(f"  Saved best pointclouds to {args.test_predict_dir}")

            # 测试结束，打印 Epoch 均值
            avg_test_loss = test_epoch_loss / len(dataloader_test)
            print(f">>> Epoch {epoch + 1} Test Summary <<<")
            print(f"    Avg Total Loss: {avg_test_loss:.4f}")
            print(f"    Avg PC Raw: {test_pc_raw_sum/len(dataloader_test):.4f}")
            print("-" * 60 + "\n")

    # ================= 6. 训练结束总结 =================
    print("\n" + "="*30 + " TRAINING FINISHED " + "="*30)
    print(f"全过程最佳 Batch 记录 (Global Best Batch Record):")
    print(f"  - Epoch: {global_best_info['epoch']}")
    print(f"  - Batch Index: {global_best_info['batch_idx']}")
    print(f"  - Total Loss (Weighted): {global_best_info['total_loss']:.6f}")
    print(f"  - Detailed Weighted Losses:")
    print(f"      PC: {global_best_info['loss_pc_weighted']:.6f}")
    print(f"      Radius: {global_best_info['loss_rad_weighted']:.6f}")
    print(f"      Theta: {global_best_info['loss_theta_weighted']:.6f}")
    print(f"  - Detailed Raw Losses:")
    print(f"      PC: {global_best_info['loss_pc_raw']:.6f}")
    print(f"      Radius: {global_best_info['loss_rad_raw']:.6f}")
    print(f"      Theta: {global_best_info['loss_theta_raw']:.6f}")
    print(f"最佳结果文件已保存在: {args.test_predict_dir}")
    print("="*80)

if __name__ == "__main__":
    main()