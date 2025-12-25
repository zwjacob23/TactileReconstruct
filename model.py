import torch
import torch.nn as nn
import numpy as np

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

class TactileTransformerMTL(nn.Module):
    def __init__(self, args):
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
        
        # ========== 任务特定的头 ==========
        self.pointcloud_head = nn.Sequential(
            nn.Linear(256, pointsNum),
            nn.ReLU(),
            nn.Linear(pointsNum, pointsNum),
            nn.ReLU(),
            nn.Linear(pointsNum, pointsNum * 3)
        )
        
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
             
        return {
            'pointcloud': pointcloud_output,
            'shape': shape_output,
            'radius_seq': radius_seq_output,
            'theta_seq': theta_seq_output
        }