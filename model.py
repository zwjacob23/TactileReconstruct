import torch
import torch.nn as nn
import numpy as np
from contrastive import ProjectionHead

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,0, :]
        return x
class FoldingNetDec(nn.Module):
    """
    基于流形的折叠解码器 (Manifold Folding Decoder)
    替代原本的全连接解码头，从 2D 栅格折叠出 3D 表面
    """
    def __init__(self, input_dim, num_points=800):
        super(FoldingNetDec, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        
        # 生成一个固定的 2D 随机网格或规则网格作为折叠基础
        # 形状: [1, 2, num_points]
        self.register_buffer('grid', torch.randn(1, 2, num_points))
        
        # Folding 1: 将 Feature + 2D Grid -> 初步 3D 坐标
        self.fold1 = nn.Sequential(
            nn.Conv1d(input_dim + 2, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 3, 1) # 输出 [B, 3, N]
        )
        
        # Folding 2: Refinement (精细化调整)
        # 将 Feature + 上一步的 3D 坐标 -> 最终 3D 坐标
        self.fold2 = nn.Sequential(
            nn.Conv1d(input_dim + 3, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 3, 1)
        )

    def forward(self, feature):
        # feature: [Batch, D] (Global Feature)
        batch_size = feature.size(0)
        
        # 1. 复制全局特征以匹配点数: [Batch, D, N]
        feature = feature.unsqueeze(2).expand(-1, -1, self.num_points)
        
        # 2. 准备 Grid: [Batch, 2, N]
        grid = self.grid.expand(batch_size, -1, -1)
        
        # 3. 第一次折叠 (拼接 Feature 和 Grid)
        x = torch.cat([feature, grid], dim=1) # [Batch, D+2, N]
        last_x = self.fold1(x) # [Batch, 3, N]
        
        # 4. 第二次折叠 (拼接 Feature 和 上一次的 3D)
        x = torch.cat([feature, last_x], dim=1) # [Batch, D+3, N]
        final_x = self.fold2(x) # [Batch, 3, N]
        
        return final_x.transpose(1, 2) # 输出 [Batch, N, 3]
class TactileTransformerMTL(nn.Module):
    def __init__(self, args, use_contrastive=False):
        super(TactileTransformerMTL, self).__init__()
        
        # 从 args 中解包参数
        input_dim1 = args.input_dim1
        input_dim2 = args.input_dim2
        num_heads = args.num_heads
        num_layers = args.num_layers
        dim_feedforward = args.dim_feedforward
        pointsNum = args.pointsNum
        seq_len = args.seq_len

        # ========== 共享编码器部分 ==========
        self.embedding1 = nn.Linear(input_dim1, 128)
        self.embedding2 = nn.Linear(input_dim2, 128)
        self.pos_encoder1 = PositionalEncoding(128)
        self.pos_encoder2 = PositionalEncoding(128)

        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads,
                                                         dim_feedforward=dim_feedforward)
        self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=num_layers)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads,
                                                         dim_feedforward=dim_feedforward)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=num_layers)

        self.fc2 = nn.Linear(150, 128)
        
        # 共享的特征提取层
        self.shared_fc = nn.Linear(384, 256)
        self.shared_fc2 = nn.Linear(256, 256)
        self.use_contrastive = use_contrastive
        if self.use_contrastive:
            # 这里的输入维度 256 必须对应 shared_features 的维度
            self.contrastive_head = ProjectionHead(input_dim=256, output_dim=128)
        
        # ========== 任务特定的头 ==========
        # self.pointcloud_head = nn.Sequential(
        #     nn.Linear(256, pointsNum),
        #     nn.ReLU(),
        #     nn.Linear(pointsNum, pointsNum),
        #     nn.ReLU(),
        #     nn.Linear(pointsNum, pointsNum * 3)
        # )
        self.pointcloud_head = FoldingNetDec(input_dim=256, num_points=pointsNum)
        self.shape_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        
        self.radius_seq_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len)
        )
        
        self.theta_seq_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len)
        )
        
        # ========== Uncertainty Weighting 参数 ==========
        self.log_var_pointcloud = nn.Parameter(torch.zeros(1))
        self.log_var_shape = nn.Parameter(torch.zeros(1))
        self.log_var_radius = nn.Parameter(torch.zeros(1))
        self.log_var_theta = nn.Parameter(torch.zeros(1))
        
        self.pointsNum = pointsNum # 保存以便 forward 使用

    def forward(self, x1, x2, x3):
        if isinstance(x3, np.ndarray):
            x3 = torch.from_numpy(x3).float()
        if x3.dim() > 2:
            x3 = x3.squeeze()
        if x3.dim() == 1:
            x3 = x3.unsqueeze(0)

        
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)

        x1 = self.pos_encoder1(x1)
        x2 = self.pos_encoder2(x2)

        x1 = self.transformer_encoder1(x1)
        x1 = x1.mean(dim=1)
        x2 = self.transformer_encoder2(x2)
        x2 = x2.mean(dim=1)

        x = torch.cat((x1, x2), dim=1)
        x3 = self.fc2(x3)
        x = torch.cat((x, x3), dim=1)
        
        shared_features = self.shared_fc(x)
        shared_features = nn.functional.relu(shared_features)
        shared_features = self.shared_fc2(shared_features)
        shared_features = nn.functional.relu(shared_features)
        
        pointcloud_output = self.pointcloud_head(shared_features)
        pointcloud_output = pointcloud_output.view(-1, self.pointsNum, 3)
        
        shape_output = self.shape_head(shared_features)
        radius_seq_output = self.radius_seq_head(shared_features)
        theta_seq_output = self.theta_seq_head(shared_features)
        output_dict = {
            'pointcloud': pointcloud_output,
            'radius_seq': radius_seq_output,
            'theta_seq': theta_seq_output,
            'shape': shape_output # 可以保留也可以不用
        }
        # ===【新增逻辑】===
        if self.use_contrastive:
            # 将共享特征投影到对比空间
            contrastive_embed = self.contrastive_head(shared_features)
            output_dict['contrastive_embed'] = contrastive_embed
            
        return output_dict
        