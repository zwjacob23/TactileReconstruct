import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys

# 引入自定义模块
from config import get_args
from utils import save_point_cloud_as_pts, ensure_dirs
from model import TactileTransformerMTL
from dataset import get_dataloaders
from pcgrad import PCGrad

# 引入 Loss
# 请确保你的 loss.py 里已经添加了 ChamferHausdorffLoss 类
from loss import uncertainty_weighted_loss, ChamferHausdorffLoss
from contrastive import SupConLoss


def main():
    # ================= 1. 初始化设置 =================
    args = get_args()
    ensure_dirs(args)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ================= 2. 准备数据 =================
    print("Loading data...")
    dataloader_train, dataloader_test = get_dataloaders(args)
    print(f"Data loaded. Train batches: {len(dataloader_train)}, Test batches: {len(dataloader_test)}")

    # ================= 3. 初始化模型与优化器 =================
    # 开启对比学习模块
    model = TactileTransformerMTL(args, use_contrastive=True).to(device)

    # --- 超参数设置 ---
    weight_contrastive = 0.1  # 对比学习权重
    weight_hd = 0.1  # Hausdorff Distance 权重 (辅助 Loss)

    if args.PCGrad:
        print(">> Using PCGrad Optimizer")
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = PCGrad(base_optimizer)
    else:
        print(">> Using Standard Adam Optimizer")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ================= 4. 定义损失函数 =================
    criterion_supcon = SupConLoss(temperature=0.07).to(device)

    # 【关键修改】 use_l1=True
    # 这样计算的是真实的物理距离 (Euclidean Distance)，而不是 MSE (Squared Distance)
    # 数值量级会回到 0.05 左右，方便与旧模型对比
    criterion_pcd = ChamferHausdorffLoss(use_l1=True)

    criterion_radius = nn.MSELoss()
    criterion_theta = nn.MSELoss()

    # ================= 5. 定义全局监控变量 =================
    # 用于记录整个训练过程中，表现最好的那个Batch (基于 Raw Chamfer Loss)
    global_best_metric = float('inf')
    global_best_info = {}

    print('Start training...')

    # ================= 6. 主循环 =================
    for epoch in range(args.epoch):

        # ----------------- 训练阶段 (Training) -----------------
        model.train()
        running_loss = 0.0
        running_pc_raw = 0.0
        running_hd_raw = 0.0
        running_rad_raw = 0.0
        running_theta_raw = 0.0

        for i, batch in enumerate(dataloader_train):
            # 解包数据
            inputs = [b.to(device) for b in batch[:3]]
            targets = [b.to(device) for b in batch[3:]]
            target_pc, target_shape, target_rad, target_theta = targets

            optimizer.zero_grad()
            outputs = model(*inputs)

            # --- 1. 计算原始 Loss (包含 HD) ---
            # criterion_pcd 返回 (Chamfer_Loss, Hausdorff_Loss)
            loss_pc_raw, loss_hd_raw = criterion_pcd(outputs['pointcloud'], target_pc)

            loss_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
            loss_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)

            # --- 2. 计算 Uncertainty Weighted Loss ---
            # HD 不参与 Uncertainty Weighting，因为它对噪声太敏感，容易导致 log_var 不稳定
            loss_pc = uncertainty_weighted_loss(loss_pc_raw, model.log_var_pointcloud)
            loss_rad = uncertainty_weighted_loss(loss_rad_raw, model.log_var_radius)
            loss_theta = uncertainty_weighted_loss(loss_theta_raw, model.log_var_theta)

            # --- 3. 计算对比学习 Loss ---
            loss_contrastive = torch.tensor(0.0).to(device)
            if 'contrastive_embed' in outputs:
                features = outputs['contrastive_embed']
                labels = target_shape
                if features.shape[0] > 1:
                    loss_contrastive = criterion_supcon(features, labels)

            # --- 4. 计算 Hausdorff 加权项 ---
            loss_hd_weighted = loss_hd_raw * weight_hd

            # --- 5. 反向传播 ---
            if args.PCGrad:
                # 构造 Loss 列表，PCGrad 会处理这些任务间的梯度冲突
                # 列表顺序：[PC_Main, Rad, Theta, PC_Aux(HD)]
                losses = [loss_pc, loss_rad, loss_theta, loss_hd_weighted]

                # 如果对比 Loss 有值，也加进去
                if loss_contrastive.item() > 0:
                    losses.append(loss_contrastive * weight_contrastive)

                optimizer.pc_backward(losses)
                optimizer.step()

                # 手动计算 total_loss 用于显示 (PCGrad 内部不返回 sum)
                total_loss = sum(losses)
            else:
                total_loss = loss_pc + loss_rad + loss_theta + \
                             loss_hd_weighted + \
                             (weight_contrastive * loss_contrastive)
                total_loss.backward()
                optimizer.step()

            # 记录数据
            running_loss += total_loss.item()
            running_pc_raw += loss_pc_raw.item()
            running_hd_raw += loss_hd_raw.item()
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
              f'| PC_Raw: {running_pc_raw / len(dataloader_train):.4f} (w={w_pc:.2f}) '
              f'| HD_Raw: {running_hd_raw / len(dataloader_train):.4f} '
              f'| Rad_Raw: {running_rad_raw / len(dataloader_train):.4f} (w={w_ra:.2f}) '
              f'| Theta_Raw: {running_theta_raw / len(dataloader_train):.4f} (w={w_th:.2f})')

        # ----------------- 测试阶段 (每10个Epoch测一次) -----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            print(f"\n>>> 开始测试 (Epoch {epoch + 1}) ...")

            test_epoch_loss = 0.0
            test_pc_raw_sum = 0.0
            test_hd_raw_sum = 0.0
            test_rad_raw_sum = 0.0
            test_theta_raw_sum = 0.0

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_test):
                    inputs = [b.to(device) for b in batch[:3]]
                    targets = [b.to(device) for b in batch[3:]]
                    target_pc, _, target_rad, target_theta = targets

                    outputs = model(*inputs)

                    # 1. 计算原始 Loss (包含 HD)
                    # 这里的 criterion_pcd 也是 use_l1=True
                    l_pc_raw, l_hd_raw = criterion_pcd(outputs['pointcloud'], target_pc)
                    l_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
                    l_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)

                    # 2. 计算加权 Loss (仅用于统计 Total Loss趋势)
                    l_pc = uncertainty_weighted_loss(l_pc_raw, model.log_var_pointcloud)
                    l_rad = uncertainty_weighted_loss(l_rad_raw, model.log_var_radius)
                    l_th = uncertainty_weighted_loss(l_theta_raw, model.log_var_theta)
                    l_hd_weighted = l_hd_raw * weight_hd

                    current_batch_total_loss_weighted = l_pc + l_rad + l_th + l_hd_weighted

                    # 记录 Epoch 统计
                    test_epoch_loss += current_batch_total_loss_weighted.item()
                    test_pc_raw_sum += l_pc_raw.item()
                    test_hd_raw_sum += l_hd_raw.item()
                    test_rad_raw_sum += l_rad_raw.item()
                    test_theta_raw_sum += l_theta_raw.item()

                    # =============== 核心：判断是否打破历史最佳记录 ===============
                    # 依然使用 l_pc_raw (Chamfer L1) 作为衡量标准
                    current_metric = l_pc_raw.item()

                    if current_metric < global_best_metric:
                        prev_best = global_best_metric
                        global_best_metric = current_metric

                        # 更新全局信息字典
                        global_best_info = {
                            'epoch': epoch + 1,
                            'batch_idx': batch_idx,
                            'best_metric_pc_raw': global_best_metric,
                            'loss_hd_raw': l_hd_raw.item(),
                            'loss_rad_raw': l_rad_raw.item(),
                            'loss_theta_raw': l_theta_raw.item()
                        }

                        # 打印详细信息
                        print(f"  [New Best Batch!] Epoch {epoch + 1} | Batch {batch_idx}")
                        print(f"  PC Raw Loss Dropped: {prev_best:.6f} -> {global_best_metric:.6f}")
                        print(
                            f"  Details: HD={l_hd_raw.item():.4f}, Rad={l_rad_raw.item():.4f}, Theta={l_theta_raw.item():.4f}")

                        # 保存当前 Batch 的点云 (取前5个样本)
                        for k in range(min(5, inputs[0].shape[0])):
                            pc_np = outputs['pointcloud'][k].cpu().numpy()
                            tgt_np = target_pc[k].cpu().numpy()

                            pred_path = os.path.join(args.test_predict_dir, f'best_sample_{k}.pts')
                            label_path = os.path.join(args.test_label_dir, f'best_sample_{k}.pts')

                            save_point_cloud_as_pts(pc_np, pred_path)
                            save_point_cloud_as_pts(tgt_np, label_path)

                        print(f"  Saved best pointclouds to {args.test_predict_dir}")

            # 测试结束，打印 Epoch 均值
            print(f">>> Epoch {epoch + 1} Test Summary <<<")
            print(f"    Avg PC Raw: {test_pc_raw_sum / len(dataloader_test):.4f}")
            print(f"    Avg HD Raw: {test_hd_raw_sum / len(dataloader_test):.4f}")
            print("-" * 60 + "\n")

    # ================= 6. 训练结束总结 =================
    print("\n" + "=" * 30 + " TRAINING FINISHED " + "=" * 30)
    if global_best_info:
        print(f"全过程最佳 Batch 记录 (Global Best Batch Record):")
        print(f"  - Epoch: {global_best_info['epoch']}")
        print(f"  - Batch Index: {global_best_info['batch_idx']}")
        print(f"  - Best PC Raw Loss: {global_best_info['best_metric_pc_raw']:.6f}")
        print(f"  - Corresponding HD Loss: {global_best_info['loss_hd_raw']:.6f}")
        print(f"最佳结果文件已保存在: {args.test_predict_dir}")
    else:
        print("未记录到任何有效测试数据。")
    print("=" * 80)


if __name__ == "__main__":
    main()