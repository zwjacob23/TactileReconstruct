import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    计算 KNN 邻居索引
    x: [B, 3, N]
    Return: [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # 取最近的 k 个邻居 (idx)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    动态构建图特征 (EdgeConv 核心)
    x: [B, C, N] (坐标或其他特征)
    Return: [B, 2*C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # 拼接：[x_i, x_j - x_i] 
    # 中心点特征 + 相对邻居的特征 -> 捕捉局部几何结构
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class RefinementNetwork(nn.Module):
    def __init__(self, num_points=800, global_feat_dim=256, k=16):
        super(RefinementNetwork, self).__init__()
        self.k = k
        self.num_points = num_points
        
        # === 1. EdgeConv 层 (处理局部几何) ===
        # 输入: [x, x-neighbor] -> 3*2 = 6 通道
        # 我们还想融入 uncertainty (1) 和 global (256)
        # 为了高效，我们先只用坐标做 KNN 提特征，后面再融合
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # === 2. 特征融合层 ===
        # EdgeConv特征(64) + Uncertainty(1) + Global(256) + 原始坐标(3)
        self.fusion_dim = 64 + 1 + global_feat_dim + 3
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.fusion_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # === 3. 偏移量预测 ===
        self.conv4 = nn.Conv1d(64, 3, 1)
        
        # 初始化为接近 0
        nn.init.uniform_(self.conv4.weight, -1e-5, 1e-5)
        nn.init.constant_(self.conv4.bias, 0)

    def forward(self, coarse_pointcloud, uncertainty_map, global_features):
        """
        coarse_pointcloud: [B, N, 3]
        uncertainty_map:   [B, N, 1]
        global_features:   [B, 256]
        """
        B, N, _ = coarse_pointcloud.shape
        
        # 1. 准备数据格式 [B, 3, N]
        pc = coarse_pointcloud.permute(0, 2, 1)
        
        # 2. 获取图特征 (Local Geometry)
        # 输入 [B, 3, N] -> 输出 [B, 6, N, k]
        graph_feat = get_graph_feature(pc, k=self.k)
        
        # 在 k 个邻居上做 Max Pooling -> [B, 64, N]
        local_feat = self.conv1(graph_feat)
        local_feat = local_feat.max(dim=-1, keepdim=False)[0]
        
        # 3. 准备其他特征
        # Global: [B, 256, 1] -> [B, 256, N]
        global_feat_expanded = global_features.unsqueeze(2).expand(-1, -1, N)
        # Uncertainty: [B, N, 1] -> [B, 1, N]
        unc_map_trans = uncertainty_map.permute(0, 2, 1)
        
        # 4. 特征拼接 (Local + Global + Uncertainty + Original Pos)
        # [B, 64+256+1+3, N]
        fusion_feat = torch.cat([local_feat, global_feat_expanded, unc_map_trans, pc], dim=1)
        
        # 5. MLP 预测残差
        x = self.conv2(fusion_feat)
        x = self.conv3(x)
        delta = self.conv4(x) # [B, 3, N]
        
        # 6. 精修
        fine_pointcloud = coarse_pointcloud + delta.permute(0, 2, 1)
        
        return fine_pointcloud