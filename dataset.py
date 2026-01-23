import torch
import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. 唯一ID参考表
ALL_UNIQUE_OBJECTS = [
    '长方体', '正方体', '高脚杯', '红酒杯', 
    '三棱柱', '圆柱体', '薯片',   '海之言', 
    '乌龙',   '七喜',   '水溶',   '怡宝'
]

class TactileDataset(Dataset):
    def __init__(self, input_data1, input_data2, input_data3, target_pointcloud, target_shape, target_radius_seq, target_theta_seq, target_obj_idx):
        self.input_data1 = input_data1
        self.input_data2 = input_data2
        self.input_data3 = input_data3
        self.target_pointcloud = target_pointcloud
        self.target_shape = target_shape
        self.target_radius_seq = target_radius_seq
        self.target_theta_seq = target_theta_seq
        self.target_obj_idx = target_obj_idx # [新增]

    def __len__(self):
        if isinstance(self.input_data1, list):
            if len(self.input_data1) == 0: return 0
            return len(self.input_data1)
        return self.input_data1.shape[0]

    def __getitem__(self, idx):
        # [关键] 这里必须返回 8 个元素
        return (
            self.input_data1[idx], 
            self.input_data2[idx], 
            self.input_data3[idx], 
            self.target_pointcloud[idx],
            self.target_shape[idx],
            self.target_radius_seq[idx],
            self.target_theta_seq[idx],
            self.target_obj_idx[idx] # <--- 缺了这个就会报错
        )

def load_data(args):
    dataset_path = os.path.join(args.root, 'dataset2', 'dataset2_addedlabel.pkl')
    ind_path = os.path.join(args.root, 'dataset2', 'hand20_ind.pkl')
    
    print(f">> Loading data from: {dataset_path}")
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)
    with open(ind_path, 'rb') as file:
        pind = pickle.load(file)

    split_mode = getattr(args, 'smode', 'object') 
    inter_mode = getattr(args, 'nmode', '3') 
    
    train_data = _init_data_container()
    test_data = _init_data_container()
    
    test_objects_list = getattr(args, 'testobj', ['七喜']) 
    target_objects = list(set(ALL_UNIQUE_OBJECTS)) 

    if split_mode == 'object':
        train_objs = [obj for obj in target_objects if obj not in test_objects_list]
        for key in train_objs:
            _extract_and_append(train_data, data, 'yzx', key, inter_mode, pind, mask=None)
        for key in test_objects_list:
            _extract_and_append(test_data, data, 'yzx', key, inter_mode, pind, mask=None)

    elif split_mode == 'round':
        for key in target_objects:
            if key not in data['yzx']: continue
            indicators = data['yzx'][key][inter_mode]['ind']
            mask_train = (indicators == 1)
            mask_test  = (indicators == 0)
            _extract_and_append(train_data, data, 'yzx', key, inter_mode, pind, mask=mask_train)
            _extract_and_append(test_data,  data, 'yzx', key, inter_mode, pind, mask=mask_test)

    elif split_mode == 'subject':
        for key in target_objects:
            if key not in data['yzx'] or key not in data['zwj']: continue
            ind_yzx = data['yzx'][key][inter_mode]['ind']
            mask_train = (ind_yzx == 1)
            _extract_and_append(train_data, data, 'yzx', key, inter_mode, pind, mask=mask_train)
            ind_zwj = data['zwj'][key][inter_mode]['ind']
            mask_test = (ind_zwj == 0)
            _extract_and_append(test_data, data, 'zwj', key, inter_mode, pind, mask=mask_test)

    return _stack_and_concat(train_data), _stack_and_concat(test_data)

def _init_data_container():
    return {'in1': [], 'in2': [], 'in3': [], 'tpc': [], 'tshape': [], 'trad': [], 'ttheta': [], 'tobj_idx': []}

def _extract_and_append(data_dict, source_data, subject, key, mode, pind, mask=None):
    if subject not in source_data or key not in source_data[subject] or mode not in source_data[subject][key]: return
    base = source_data[subject][key][mode]
    
    def get_subset(arr):
        if mask is not None: return arr[mask]
        return arr

    pressure = get_subset(base['pressure'])[:, :, pind]
    joints = get_subset(base['joints'])
    distance = get_subset(base['distance'])
    pointcloud = get_subset(base['pointcloud'])
    shape = get_subset(base['shape'])
    radius_seq = get_subset(base['radius_seq'])
    theta_seq = get_subset(base['theta_seq'])

    # [关键] 生成 ID
    if key in ALL_UNIQUE_OBJECTS:
        unique_id = ALL_UNIQUE_OBJECTS.index(key)
    else:
        unique_id = -1
    
    num_samples = len(pressure)
    obj_idx_arr = np.full((num_samples,), unique_id, dtype=np.int64)

    data_dict['in1'].append(torch.tensor(pressure, dtype=torch.float32))
    data_dict['in2'].append(torch.tensor(joints, dtype=torch.float32))
    data_dict['in3'].append(torch.tensor(distance, dtype=torch.float32))
    data_dict['tpc'].append(torch.tensor(pointcloud, dtype=torch.float32))
    data_dict['tshape'].append(torch.tensor(shape, dtype=torch.long))
    data_dict['trad'].append(torch.tensor(radius_seq, dtype=torch.float32))
    data_dict['ttheta'].append(torch.tensor(theta_seq, dtype=torch.float32))
    data_dict['tobj_idx'].append(torch.tensor(obj_idx_arr, dtype=torch.long)) # <--- 关键 Append

def _stack_and_concat(data_dict):
    result = []
    # [关键] 列表里必须有8个key
    keys = ['in1', 'in2', 'in3', 'tpc', 'tshape', 'trad', 'ttheta', 'tobj_idx']
    if not data_dict['in1']: return [torch.empty(0) for _ in keys]
    for k in keys:
        result.append(torch.cat(data_dict[k], dim=0))
    return result

def get_dataloaders(args):
    train_tensors, test_tensors = load_data(args)
    train_dataset = TactileDataset(*train_tensors)
    test_dataset = TactileDataset(*test_tensors)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader