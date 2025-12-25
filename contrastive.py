import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    对比学习专用的投影头 (Projection Head)
    将 backbone 提取的特征映射到对比学习的嵌入空间
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        # 对比学习的关键：输出必须进行 L2 归一化，投影到单位球面上
        return F.normalize(x, dim=1)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss: https://arxiv.org/pdf/2004.11362.pdf
    允许一个 batch 里有多个同类样本（正样本对）
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, dim] - 经过 ProjectionHead 归一化后的特征
            labels: [batch_size] - 对应的形状标签 (0, 1, 2...)
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. 计算相似度矩阵 (Cosine Similarity)
        # features 已经是归一化的，所以 dot product 等于 cosine similarity
        similarity_matrix = torch.matmul(features, features.T) # [batch, batch]

        # 2. 构建 Mask
        # labels.unsqueeze(1) -> [B, 1], labels.unsqueeze(0) -> [1, B]
        # labels_match 矩阵中，(i, j) 为 True 表示 i 和 j 是同类
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
            
        mask = torch.eq(labels, labels.T).float().to(device)

        # 3. 移除对角线 (自己与自己的对比没有意义)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask # 仅保留同类且非自身的正样本位置

        # 4. 计算 Logits
        # 除以温度系数
        logits = similarity_matrix / self.temperature
        
        # 为了数值稳定性，减去每行的最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 5. 计算 Loss
        # 分母：所有样本的指数和 (除了自己)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # 分子：仅计算正样本对的 log_prob
        # mean_log_prob_pos: 每个样本与其所有正样本对的平均 log_likelihood
        # mask.sum(1) 是每个样本拥有的正样本数量
        mask_sum = mask.sum(1)
        
        # 处理没有任何正样本的情况 (防止除以0)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 最终 loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        
        return loss