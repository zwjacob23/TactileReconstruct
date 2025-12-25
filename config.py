import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Tactile Transformer MTL Training")
    
    # 路径设置
    parser.add_argument('--root', type=str, default='../data', help='数据根目录')
    parser.add_argument('--train_predict_dir', type=str, default='./output_pointclouds/train_2/predict')
    parser.add_argument('--train_label_dir', type=str, default='./output_pointclouds/train_2/label')
    parser.add_argument('--test_predict_dir', type=str, default='./output_pointclouds/test_2/predict')
    parser.add_argument('--test_label_dir', type=str, default='./output_pointclouds/test_2/label')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # 模型参数
    parser.add_argument('--pointsNum', type=int, default=800)
    parser.add_argument('--input_dim1', type=int, default=361)
    parser.add_argument('--input_dim2', type=int, default=20)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=20)
    
    return parser.parse_args()