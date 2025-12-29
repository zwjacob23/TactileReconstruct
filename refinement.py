import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementNetwork(nn.Module):
    def __init__(self, num_points=800, global_feat_dim=256):
        super(RefinementNetwork, self).__init__()
        
        # ===【修改点】输入维度变大 ===
        # 3 (坐标) + 1 (不确定性) + 256 (全局特征) = 260
        self.input_dim = 3 + 1 + global_feat_dim
        
        self.conv1 = nn.Conv1d(self.input_dim, 128, 1) # 加宽网络
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, 3, 1) # 输出偏移量

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        nn.init.uniform_(self.conv3.weight, -1e-5, 1e-5)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, coarse_pointcloud, uncertainty_map, global_features):
        """
        coarse_pointcloud: [B, N, 3]
        uncertainty_map:   [B, N, 1]
        global_features:   [B, 256] (来自 Transformer 的共享特征)
        """
        B, N, _ = coarse_pointcloud.shape
        
        # 1. 扩展全局特征到每个点
        # [B, 256] -> [B, 256, 1] -> [B, 256, N]
        global_feat_expanded = global_features.unsqueeze(2).expand(-1, -1, N)
        
        # 2. 准备局部特征
        # [B, N, 4] -> [B, 4, N]
        local_feat = torch.cat([coarse_pointcloud, uncertainty_map], dim=-1).permute(0, 2, 1)
        
        # 3. 拼接全局和局部 [B, 260, N]
        x = torch.cat([local_feat, global_feat_expanded], dim=1)
        
        # 4. 卷积精修
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        delta = self.conv3(x)
        
        # 5. 输出
        fine_pointcloud = coarse_pointcloud + delta.permute(0, 2, 1)
        return fine_pointcloud