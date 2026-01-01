import torch
import torch.nn as nn

def uncertainty_weighted_loss(task_loss, log_var):
    """
    计算不确定性加权损失
    Args:
        task_loss: 任务的原始损失值（标量）
        log_var: 可学习的 log variance 参数 (log(sigma^2))
    """
    precision = torch.exp(-log_var)  # 1/sigma^2
    log_var_clamped = torch.clamp(log_var, min=-2.0, max=10.0)

    precision = torch.exp(-log_var_clamped)
    weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var_clamped
    return weighted_loss

class ChamferHausdorffLoss(nn.Module):
    def __init__(self, use_l1=False):
        super(ChamferHausdorffLoss, self).__init__()
        self.use_l1 = use_l1

    def forward(self, pred, gt):
        """
        pred: [B, N, 3]
        gt:   [B, M, 3]
        """
        # 1. 计算距离矩阵 (B, N, M)
        # 展开为 (B, N, 1, 3) 和 (B, 1, M, 3) 进行广播计算
        # 注意：如果点数特别多(>2000)，这步可能会显存爆炸，800点完全没问题
        pred_expand = pred.unsqueeze(2)
        gt_expand = gt.unsqueeze(1)

        # 计算欧氏距离平方
        # dist_matrix: [B, N, M]
        dist_matrix = torch.sum((pred_expand - gt_expand) ** 2, dim=-1)

        if self.use_l1:
            dist_matrix = torch.sqrt(dist_matrix + 1e-8)

        # 2. 找到最近邻
        # dist1: pred中每个点到gt最近点的距离 [B, N]
        # dist2: gt中每个点到pred最近点的距离 [B, M]
        min_dist1, _ = torch.min(dist_matrix, dim=2)
        min_dist2, _ = torch.min(dist_matrix, dim=1)

        # 3. 计算 Chamfer Distance (平均距离)
        chamfer_loss = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)
        chamfer_loss = chamfer_loss.mean()  # Batch mean

        # 4. 计算 Hausdorff Distance (最大距离)
        # 这种写法是 "Average Hausdorff"，即双向最大值的平均，比纯最大值稳定
        # 也可以用 torch.max(min_dist1, dim=1)[0] 取绝对最大值
        max_dist1, _ = torch.max(min_dist1, dim=1)
        max_dist2, _ = torch.max(min_dist2, dim=1)
        hausdorff_loss = torch.mean(max_dist1) + torch.mean(max_dist2)

        return chamfer_loss, hausdorff_loss
## hhh
