import torch

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



## hhh
