import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import random
import numpy as np
from contrastive import SupConLoss
from chanmferdistance import ChamferDistance
from pcgrad import PCGrad
from config import get_args
from utils import save_point_cloud_as_pts, ensure_dirs
from loss import uncertainty_weighted_loss
from model import TactileTransformerMTL
from dataset import get_dataloaders

# =========================================================
# 【新增】 噪声注入函数
# =========================================================
def apply_gaussian_noise(tensor, noise_level=0.0):
    """
    给输入张量添加高斯噪声 (假设输入已经归一化到 [-1, 1] 或 [0, 1])
    noise_level: 标准差, e.g., 0.01 代表 1% 的噪声
    """
    if noise_level <= 0:
        return tensor
    # 生成噪声并叠加
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise

def setup_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> [Info] Random seed set to {seed}. Results will be reproducible.")

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

    # ================= 4. 定义全局监控变量 =================
    global_best_batch_loss = float('inf')  
    global_best_info = {} # 省略初始化内容，保持原样即可

    print('Start training...')
    
    # ================= 5. 主循环 =================
    for epoch in range(args.epoch):
        
        # ----------------- 训练阶段 (Training) -----------------
        # 【策略】训练阶段保持 Clean (无噪声)，利用正则项学习鲁棒特征
        model.train()
        running_loss = 0.0
        running_pc_raw = 0.0
        running_rad_raw = 0.0
        running_theta_raw = 0.0
        
        for i, batch in enumerate(dataloader_train):
            inputs = [b.to(device) for b in batch[:3]]
            targets = [b.to(device) for b in batch[3:]]
            target_pc, target_shape, target_rad, target_theta = targets
            
            optimizer.zero_grad()
            outputs = model(*inputs)
            
            # 1. 计算原始 Loss
            loss_pc_raw = chamfer_distance(outputs['pointcloud'], target_pc)
            loss_rad_raw = criterion_radius(outputs['radius_seq'], target_rad)
            loss_theta_raw = criterion_theta(outputs['theta_seq'], target_theta)

            # 【修改】删除了 * 1000 的操作，回归标准归一化尺度
            # loss_pc_raw = loss_pc_raw * 1000  <-- 已删除

            # 2. 计算 Uncertainty Weighted Loss
            if epoch < 50:
                loss_pc = loss_pc_raw * 20.0 
                loss_rad = loss_rad_raw * 1.0
                loss_theta = loss_theta_raw * 1.0
            else:
                # 这里的 loss 是小数，s 会变负来增大权重，Ours 会防止它变得过负 (Anti-Dominance)
                loss_pc = uncertainty_weighted_loss(loss_pc_raw, model.log_var_pointcloud, regularization_weight=0.2)
                loss_rad = uncertainty_weighted_loss(loss_rad_raw, model.log_var_radius, regularization_weight=0.1)
                loss_theta = uncertainty_weighted_loss(loss_theta_raw, model.log_var_theta, regularization_weight=0.1)
            
            # 3. 对比学习 & 反向传播 (保持不变)
            loss_contrastive = torch.tensor(0.0).to(device)
            if 'contrastive_embed' in outputs and features.shape[0] > 1:
                 # ... (省略具体代码，保持原样) ...
                 pass 

            if args.PCGrad:
                losses = [loss_pc, loss_rad, loss_theta]
                # ... (PCGrad逻辑) ...
                optimizer.pc_backward(losses)
                optimizer.step()
            else:
                total_loss = loss_pc + loss_rad + loss_theta # + contrastive ...
                total_loss.backward()
                optimizer.step()
            
            # 记录数据 (注意：PCGrad 模式下 total_loss 计算略有不同，这里简化展示)
            running_loss += total_loss.item() if not args.PCGrad else sum([l.item() for l in losses])
            running_pc_raw += loss_pc_raw.item()
            running_rad_raw += loss_rad_raw.item()
            running_theta_raw += loss_theta_raw.item()

        # 打印训练 Epoch 信息
        avg_train_loss = running_loss / len(dataloader_train)
        with torch.no_grad():
            w_pc = torch.exp(-model.log_var_pointcloud).item()
            w_ra = torch.exp(-model.log_var_radius).item()
            w_th = torch.exp(-model.log_var_theta).item()

        print(f'Epoch [{epoch + 1}/{args.epoch}] Train Loss: {avg_train_loss:.4f} '
              f'| PC: {running_pc_raw/len(dataloader_train):.4f} (w={w_pc:.2f}) '
              f'| Rad: {running_rad_raw/len(dataloader_train):.4f} (w={w_ra:.2f}) '
              f'| Theta: {running_theta_raw/len(dataloader_train):.4f} (w={w_th:.2f})')

        # ----------------- 测试阶段 (Robustness Test Loop) -----------------
        # 【修改】每10轮，或者最后一轮，进行一次完整的多级噪声测试
        if (epoch + 1) % 10 == 0:
            model.eval()
            print(f"\n>>> 开始多级抗噪测试 (Epoch {epoch + 1}) ...")
            
            # 定义噪声等级：0=Clean, 0.01=Low, 0.03=Med, 0.05=High, 0.1=Extreme
            noise_levels = [0.0, 0.01, 0.03, 0.05, 0.10]
            results_list = [] # 用于存结果

            for noise_lvl in noise_levels:
                test_loss_sum = 0.0
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader_test):
                        inputs = [b.to(device) for b in batch[:3]]
                        targets = [b.to(device) for b in batch[3:]]
                        target_pc, _, target_rad, target_theta = targets
                        
                        # --- 【关键】注入噪声 ---
                        # 假设 inputs[0] 是主要的触觉/点云输入
                        inputs[0] = apply_gaussian_noise(inputs[0], noise_lvl)
                        # 如果 inputs[1] 也是需要加噪的模态，也加上
                        # inputs[1] = apply_gaussian_noise(inputs[1], noise_lvl)
                        
                        outputs = model(*inputs)
                        
                        # --- 计算指标 ---
                        # 因为训练和输入都是归一化的，直接算 CD 就是标准指标
                        metric_cd = chamfer_distance(outputs['pointcloud'], target_pc)
                        test_loss_sum += metric_cd.item()
                        
                        # (可选) 如果是 Clean 模式 (noise=0)，可以顺便更新“最佳模型”逻辑
                        if noise_lvl == 0.0 and metric_cd.item() < global_best_batch_loss:
                            global_best_batch_loss = metric_cd.item()
                            # ... (更新 global_best_info，代码保持你原来的即可) ...
                
                avg_cd = test_loss_sum / len(dataloader_test)
                results_list.append(avg_cd)
                print(f"  [Noise {noise_lvl}] Avg CD: {avg_cd:.6f}")

            # 打印方便画图的列表
            print(f"\n>>> Epoch {epoch+1} Robustness Summary (Copy to plot) <<<")
            print(f"X (Noise): {noise_levels}")
            print(f"Y (CD):    {results_list}")
            print("-" * 60 + "\n")

    # ... (训练结束总结代码保持不变) ...

if __name__ == "__main__":
    main()