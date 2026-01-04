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

# === [核心修改] FoldingNet 解码器 (兼容旧版 PyTorch) ===
class FoldingNetDecoder(nn.Module):
    def __init__(self, input_dim=256, num_points=800):
        super(FoldingNetDecoder, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        
        # 1. 生成固定的 2D 网格 (Grid)
        # 为了凑齐 800 个点，我们使用 25 * 32 = 800 的网格
        range_x = torch.linspace(-1.0, 1.0, steps=25) 
        range_y = torch.linspace(-1.0, 1.0, steps=32)
        
        # [修改点] 移除了 indexing='ij'，兼容旧版本 PyTorch
        # 旧版本默认就是 'ij' 模式
        grid_x, grid_y = torch.meshgrid(range_x, range_y)
        
        self.grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2) # [800, 2]
        
        # 2. 折叠操作 (Folding MLP)
        # 输入: Global Feat (256) + Grid Coord (2) = 258
        self.folding1 = nn.Sequential(
            nn.Linear(input_dim + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # 输出 3D 坐标
        )

    def forward(self, x):
        # x: [Batch, 256] (全局特征)
        batch_size = x.size(0)
        
        # 1. 复制 Grid 到每个 Batch
        # grid: [Batch, N, 2]
        # 注意：self.grid 需要移动到和 x 相同的 device 上
        grid = self.grid.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 复制 Global Feature 到每个点
        # x_expand: [Batch, N, 256]
        x_expand = x.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # 3. 拼接
        # cat: [Batch, N, 258]
        cat_feat = torch.cat([x_expand, grid], dim=-1)
        
        # 4. 折叠
        # point_cloud: [Batch, N, 3]
        point_cloud = self.folding1(cat_feat)
        
        return point_cloud

class TactileTransformerMTL(nn.Module):
    def __init__(self, args, use_contrastive=False):
        super(TactileTransformerMTL, self).__init__()
        
        # 参数解包
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
        
        # 共享特征层
        self.shared_fc = nn.Linear(384, 256)
        self.shared_fc2 = nn.Linear(256, 256)
        
        self.use_contrastive = use_contrastive
        if self.use_contrastive:
            self.contrastive_head = ProjectionHead(input_dim=256, output_dim=128)
        
        # ========== [修改] 任务头：使用 FoldingNet ==========
        # 替换掉了原来的 MLP Head，改用 FoldingNetDecoder
        self.pointcloud_decoder = FoldingNetDecoder(input_dim=256, num_points=pointsNum)
        
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
        
        # 任务权重参数
        self.log_var_pointcloud = nn.Parameter(torch.zeros(1))
        self.log_var_shape = nn.Parameter(torch.zeros(1))
        self.log_var_radius = nn.Parameter(torch.zeros(1))
        self.log_var_theta = nn.Parameter(torch.zeros(1))
        
        self.pointsNum = pointsNum

    def forward(self, x1, x2, x3):
        # 输入处理
        if isinstance(x3, np.ndarray):
            x3 = torch.from_numpy(x3).float()
        if x3.dim() > 2:
            x3 = x3.squeeze()
        if x3.dim() == 1:
            x3 = x3.unsqueeze(0)

        # Encoder 前向传播
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
        
        # 共享特征提取
        shared_features = self.shared_fc(x)
        shared_features = nn.functional.relu(shared_features)
        shared_features = self.shared_fc2(shared_features)
        shared_features = nn.functional.relu(shared_features)
        
        # [修改] 使用 Decoder 生成点云
        pointcloud_output = self.pointcloud_decoder(shared_features)
        # pointcloud_output 已经是 [B, 800, 3]
        
        shape_output = self.shape_head(shared_features)
        radius_seq_output = self.radius_seq_head(shared_features)
        theta_seq_output = self.theta_seq_head(shared_features)
        
        output_dict = {
            'pointcloud': pointcloud_output,
            'radius_seq': radius_seq_output,
            'theta_seq': theta_seq_output,
            'shape': shape_output
        }
        
        if self.use_contrastive:
            contrastive_embed = self.contrastive_head(shared_features)
            output_dict['contrastive_embed'] = contrastive_embed
            
        return output_dict