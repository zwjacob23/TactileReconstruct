import torch
import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TactileDataset(Dataset):
    def __init__(self, input_data1, input_data2, input_data3, target_pointcloud, target_shape, target_radius_seq, target_theta_seq):
        self.input_data1 = input_data1
        self.input_data2 = input_data2
        self.input_data3 = input_data3
        self.target_pointcloud = target_pointcloud
        self.target_shape = target_shape
        self.target_radius_seq = target_radius_seq
        self.target_theta_seq = target_theta_seq

    def __len__(self):
        return self.input_data1.shape[0]

    def __getitem__(self, idx):
        return (
            self.input_data1[idx], 
            self.input_data2[idx], 
            self.input_data3[idx], 
            self.target_pointcloud[idx],
            self.target_shape[idx],
            self.target_radius_seq[idx],
            self.target_theta_seq[idx]
        )

def load_data(args):
    """
    负责读取pkl文件，并进行数据清洗和划分
    """
    dataset_path = os.path.join(args.root,  'dataset2_addedlabel.pkl')
    ind_path = os.path.join(args.root, 'hand20_ind.pkl')

    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)
    with open(ind_path, 'rb') as file:
        pind = pickle.load(file)

    # 数据容器
    train_data = {'in1': [], 'in2': [], 'in3': [], 'tpc': [], 'tshape': [], 'trad': [], 'ttheta': []}
    test_data = {'in1': [], 'in2': [], 'in3': [], 'tpc': [], 'tshape': [], 'trad': [], 'ttheta': []}

    # 划分逻辑：以物体划分 (你目前使用的逻辑)
    test_objects = ['长方体']
    # 注意：这里建议把所有物体名放到列表中，不要硬编码
    all_objects = ['水溶','七喜','海之言','怡宝','高脚杯','三棱柱','乌龙','正方体','薯片','圆柱体','红酒杯']
    
    # 加载训练集物体
    for key in all_objects:
        _append_to_dict(train_data, data, key, pind)

    # 加载测试集物体
    for key in test_objects:
        _append_to_dict(test_data, data, key, pind)

    # 转换为 Tensor 并拼接
    train_tensors = _stack_and_concat(train_data)
    test_tensors = _stack_and_concat(test_data)
    
    return train_tensors, test_tensors

def _append_to_dict(data_dict, source_data, key, pind):
    """辅助函数：从源数据提取并append到字典"""
    # 这里假设源数据的结构是 data['yzx'][key]['3']...
    # 如果有不同的结构，需要在这里调整
    base = source_data['yzx'][key]['3']
    
    data_dict['in1'].append(torch.tensor(base['pressure'][:,:,pind], dtype=torch.float32))
    data_dict['in2'].append(torch.tensor(base['joints'], dtype=torch.float32))
    data_dict['in3'].append(torch.tensor(base['distance'], dtype=torch.float32))
    data_dict['tpc'].append(torch.tensor(base['pointcloud'], dtype=torch.float32))
    data_dict['tshape'].append(torch.tensor(base['shape'], dtype=torch.long))
    data_dict['trad'].append(torch.tensor(base['radius_seq'], dtype=torch.float32))
    data_dict['ttheta'].append(torch.tensor(base['theta_seq'], dtype=torch.float32))

def _stack_and_concat(data_dict):
    """辅助函数：将列表拼接为Tensor"""
    return [torch.cat(data_dict[k], dim=0) for k in data_dict]

def get_dataloaders(args):
    train_tensors, test_tensors = load_data(args)
    
    train_dataset = TactileDataset(*train_tensors)
    test_dataset = TactileDataset(*test_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_loader, test_loader