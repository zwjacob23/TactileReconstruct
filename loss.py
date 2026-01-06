import torch
import torch

def uncertainty_weighted_loss(task_loss, log_var, regularization_weight=0.1):
    """
    计算带有先验正则化的不确定性加权损失 (Prior-Regularized Uncertainty Weighting)
    
    Args:
        task_loss: 任务的原始损失值（标量）
        log_var: 可学习的 log variance 参数 (log(sigma^2))
        regularization_weight: 正则化强度 (lambda)，用于防止不确定性无限增大 (Task Collapse)
    """
    # 1. 安全截断，防止数值爆炸
    # 原始范围 [-2, 10]，这里保持一致
    log_var_clamped = torch.clamp(log_var, min=-2.0, max=10.0)
    
    # 2. 计算精度 (Precision = 1 / sigma^2)
    # precision 代表了该任务的"权重"
    precision = torch.exp(-log_var_clamped)
    
    # 3. 经典的 Kendall et al. Loss
    # L = 0.5 * (1/sigma^2) * Loss + 0.5 * log(sigma^2)
    standard_loss = 0.5 * precision * task_loss + 0.5 * log_var_clamped
    
    # 4. 【创新点】：先验正则化 (Prior Regularization)
    # 这是一个 L2 惩罚项，鼓励 log_var 接近 0 (即鼓励 sigma^2 接近 1，权重接近 1)
    # 物理含义：我们假设任务权重的先验分布是以 1 为中心的，防止网络过度"偷懒"
    prior_reg = regularization_weight * (log_var_clamped ** 2)
    
    # 总损失
    total_loss = standard_loss + prior_reg
    
    return total_loss
