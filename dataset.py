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
        # 增加鲁棒性，防止空数据报错
        if isinstance(self.input_data1, list):
            if len(self.input_data1) == 0: return 0
            return len(self.input_data1)
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
    根据 args.split_mode 和 args.interaction_mode 灵活加载数据
    """
    dataset_path = os.path.join(args.root, 'dataset2', 'dataset2_addedlabel.pkl')
    ind_path = os.path.join(args.root, 'dataset2', 'hand20_ind.pkl')
    # args.split_mode = 'round'      # 改这里: 'object', 'round', 'subject'
    # args.interaction_mode = '3'
    print(f">> Loading data from: {dataset_path}")
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)
    with open(ind_path, 'rb') as file:
        pind = pickle.load(file)

    # === 1. 获取配置参数 (优先从args获取，没有则使用默认值) ===
    # split_mode: 'object' (物体), 'round' (回合), 'subject' (受试者)
    split_mode = getattr(args, 'smode', 'object') 
    # interaction_mode: '2', '3', '5' (点数)
    inter_mode = getattr(args, 'nmode', '3') 
    
    print(f">> Dataset Split Mode: [{split_mode}] | Interaction Mode: [{inter_mode}]")

    # 数据容器
    train_data = _init_data_container()
    test_data = _init_data_container()

    # 物体列表
    all_objects = ['水溶','长方体','海之言','怡宝','高脚杯','三棱柱','乌龙','正方体','薯片','圆柱体','红酒杯', '七喜']
    # 如果是跨物体实验，这些是测试集物体；如果是其他实验，这些通常包含在所有数据中
    test_objects_list = getattr(args, 'testobj', ['七喜']) 

    # === 2. 根据模式进行划分 ===
    
    # ------------------ 模式 A: 以物体划分 (Cross-Object) ------------------
    if split_mode == 'object':
        # 训练集：all_objects 中排除掉 test_objects
        train_objs = [obj for obj in all_objects if obj not in test_objects_list]
        
        for key in train_objs:
            # 训练集：yzx, 无 mask (取全部)
            _extract_and_append(train_data, data, 'yzx', key, inter_mode, pind, mask=None)
            
        for key in test_objects_list:
            # 测试集：yzx, 无 mask (取全部)
            _extract_and_append(test_data, data, 'yzx', key, inter_mode, pind, mask=None)

    # ------------------ 模式 B: 以交互回合划分 (Cross-Round) ------------------
    elif split_mode == 'round':
        # 遍历所有物体 (或者你可以只选部分物体，这里默认用 all_objects + test_objects_list)
        # 注意：你的原始代码里 yqs 包含了所有物体，所以这里我们合并列表去重
        target_objects = list(set(all_objects + test_objects_list))
        
        for key in target_objects:
            if key not in data['yzx']: continue
            
            # 获取 mask
            # ind == 1 为训练, ind == 0 为测试
            indicators = data['yzx'][key][inter_mode]['ind']
            mask_train = (indicators == 1)
            mask_test  = (indicators == 0)
            
            _extract_and_append(train_data, data, 'yzx', key, inter_mode, pind, mask=mask_train)
            _extract_and_append(test_data,  data, 'yzx', key, inter_mode, pind, mask=mask_test)

    # ------------------ 模式 C: 以受试者划分 (Cross-Subject) ------------------
    elif split_mode == 'subject':
        target_objects = list(set(all_objects + test_objects_list))
        
        for key in target_objects:
            if key not in data['yzx'] or key not in data['zwj']: continue
            
            # 训练集：来自 'yzx', ind == 1
            ind_yzx = data['yzx'][key][inter_mode]['ind']
            mask_train = (ind_yzx == 1)
            _extract_and_append(train_data, data, 'yzx', key, inter_mode, pind, mask=mask_train)
            
            # 测试集：来自 'zwj', ind == 0
            ind_zwj = data['zwj'][key][inter_mode]['ind']
            mask_test = (ind_zwj == 0)
            _extract_and_append(test_data, data, 'zwj', key, inter_mode, pind, mask=mask_test)

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # === 3. 堆叠与拼接 ===
    train_tensors = _stack_and_concat(train_data)
    test_tensors = _stack_and_concat(test_data)
    
    return train_tensors, test_tensors

def _init_data_container():
    return {'in1': [], 'in2': [], 'in3': [], 'tpc': [], 'tshape': [], 'trad': [], 'ttheta': []}

def _extract_and_append(data_dict, source_data, subject, key, mode, pind, mask=None):
    """
    通用提取函数
    Args:
        mask: 布尔数组或None。如果是None，提取所有数据；否则只提取 mask 为 True 的数据。
    """
    # 检查键值是否存在
    if subject not in source_data: return
    if key not in source_data[subject]: return
    if mode not in source_data[subject][key]:
        # print(f"Warning: Mode {mode} not found for {subject}-{key}")
        return

    base = source_data[subject][key][mode]
    
    # 定义提取逻辑：先根据 mask 筛选 batch，再处理特征
    def get_subset(arr):
        if mask is not None:
            return arr[mask]
        return arr

    # 1. Pressure: [Batch, Time, Channel] -> 筛选 Batch -> 筛选 Channel (pind)
    pressure = get_subset(base['pressure'])
    pressure = pressure[:, :, pind] # 特征筛选
    
    # 2. 其他数据
    joints = get_subset(base['joints'])
    distance = get_subset(base['distance'])
    pointcloud = get_subset(base['pointcloud'])
    shape = get_subset(base['shape'])
    radius_seq = get_subset(base['radius_seq'])
    theta_seq = get_subset(base['theta_seq'])

    # Append 到列表 (转换为 Tensor)
    data_dict['in1'].append(torch.tensor(pressure, dtype=torch.float32))
    data_dict['in2'].append(torch.tensor(joints, dtype=torch.float32))
    data_dict['in3'].append(torch.tensor(distance, dtype=torch.float32))
    data_dict['tpc'].append(torch.tensor(pointcloud, dtype=torch.float32))
    data_dict['tshape'].append(torch.tensor(shape, dtype=torch.long))
    data_dict['trad'].append(torch.tensor(radius_seq, dtype=torch.float32))
    data_dict['ttheta'].append(torch.tensor(theta_seq, dtype=torch.float32))

def _stack_and_concat(data_dict):
    """将列表拼接为Tensor，处理空列表情况"""
    result = []
    # 按照固定的顺序 keys
    keys = ['in1', 'in2', 'in3', 'tpc', 'tshape', 'trad', 'ttheta']
    
    # 检查是否有数据
    if not data_dict['in1']:
        print("Warning: No data found for this split configuration!")
        # 返回空 Tensor 防止报错 (或者在 Dataset __len__ 里处理)
        return [torch.empty(0) for _ in keys]

    for k in keys:
        # cat dim=0
        result.append(torch.cat(data_dict[k], dim=0))
    return result

def get_dataloaders(args):
    train_tensors, test_tensors = load_data(args)
    
    train_dataset = TactileDataset(*train_tensors)
    test_dataset = TactileDataset(*test_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_loader, test_loader