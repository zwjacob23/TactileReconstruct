import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import platform

# ================= 配置区域 (请根据实际情况修改) =================
# 1. 你的数据集路径
DATASET_PATH = 'D:/research/Reconstruction-master/data/dataset2/prudataset.pkl' 

# 2. 交互模式 (必须保持一致，建议用 '5' 或 '3'，数据量大一点分布更明显)
INTERACTION_MODE = '3'                

# 3. 定义两个受试者 (对应 pickle 文件里的 key)
# 假设 'zwj' 是手小的 (Subject A/Train)，'yzx' 是手大的 (Subject B/Test)
# 如果你不确定 key 是什么，代码运行第一步会打印出来，你再改
SUBJECTS = ['zwj', 'yzx'] 
SUBJECT_LABELS = ['Subject A (Train)', 'Subject B (Test)'] # 图例显示的名称

# 4. 共同的物体列表 (只对比两个人都抓过的物体，确保分布差异来自“人”而不是“物”)
TARGET_OBJECTS = [
    'SodaBtl', 'Cylinder', 'Cuboid', 'ChipTube', 
    'WineGlass', 'SportBtl', 'Triprism', 'Goblet',
    'JuiceBtl', 'Cube', 'WaterBtl', 'TeaBtl'
]
# ==============================================================

# 设置绘图风格 (学术论文风)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # 英文论文标准字体
plt.rcParams['axes.unicode_minus'] = False 

def load_user_data(pkl_path, user, mode, target_objs):
    """提取单个用户的所有指定物体特征"""
    if not os.path.exists(pkl_path):
        print(f"❌ 错误: 找不到文件 {pkl_path}")
        return None, None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if user not in data:
        print(f"❌ 错误: 用户 '{user}' 不在数据集中。可用 Key: {list(data.keys())}")
        return None, None

    user_data = data[user]
    pressure_list = []
    joint_list = []
    
    # 遍历物体
    for obj_name in target_objs:
        if obj_name not in user_data:
            continue
        if mode not in user_data[obj_name]:
            continue

        # 获取数据 (Batch, Seq, Dim)
        raw_p = np.array(user_data[obj_name][mode]['pressure'])
        raw_j = np.array(user_data[obj_name][mode]['joints'])

        # Flatten: (Batch, Seq*Dim) - 展平以便进行 t-SNE
        # 关节角度: (Batch, 150*20) -> (Batch, 3000)
        p_flat = raw_p.reshape(raw_p.shape[0], -1)
        j_flat = raw_j.reshape(raw_j.shape[0], -1)

        pressure_list.append(p_flat)
        joint_list.append(j_flat)

    if len(pressure_list) == 0:
        return None, None

    # 拼接该用户所有物体的数据
    X_p = np.concatenate(pressure_list, axis=0)
    X_j = np.concatenate(joint_list, axis=0)
    
    return X_p, X_j

def run_domain_shift_analysis():
    print(f"正在加载数据集: {DATASET_PATH}...")
    
    # 容器
    all_joints = []
    all_pressures = []
    domain_labels = [] # 存储 "Subject A", "Subject B"
    
    # 1. 加载两个用户的数据
    for sub_key, sub_label in zip(SUBJECTS, SUBJECT_LABELS):
        print(f"正在提取用户 [{sub_key}] 的特征...")
        X_p, X_j = load_user_data(DATASET_PATH, sub_key, INTERACTION_MODE, TARGET_OBJECTS)
        
        if X_p is None:
            print(f"跳过用户 {sub_key} (无数据)")
            continue
            
        all_pressures.append(X_p)
        all_joints.append(X_j)
        # 为当前数据生成标签
        domain_labels.extend([sub_label] * X_p.shape[0])
        
    # 2. 合并数据
    X_final_joints = np.concatenate(all_joints, axis=0)
    X_final_pressure = np.concatenate(all_pressures, axis=0)
    y_final_labels = np.array(domain_labels)
    
    print(f"\n数据准备完毕: 总样本数 {len(y_final_labels)}")
    print(f"关节特征维度: {X_final_joints.shape}")
    
    # 3. 绘制 t-SNE (重点画关节角度)
    plot_distribution(X_final_joints, y_final_labels, 
                     title="(a) Domain Shift in Proprioception (Joint Angles)", 
                     filename="domain_shift_joints.png")
    
    # 4. (可选) 绘制 压力分布
    # 压力分布可能重叠较多，因为不同手抓同一个物体，接触面可能相似，也可能不同
    plot_distribution(X_final_pressure, y_final_labels, 
                     title="Domain Shift in Tactile Pressure", 
                     filename="domain_shift_pressure.png")

def plot_distribution(X, y, title, filename):
    print(f"正在计算 t-SNE: {title} ...")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 预处理 (加速)
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)
        
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X_scaled)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    
    # 使用对比度高的颜色: 蓝色 vs 橙色/红色
    palette = {'Subject A (Train)': '#2980b9', 'Subject B (Test)': '#e67e22'}
    # 如果你的标签不是这两个，seaborn会自动选色
    
    sns.scatterplot(
        x=X_emb[:, 0], y=X_emb[:, 1],
        hue=y,
        style=y, # 不同形状增加区分度
        palette=['#2980b9', '#e67e22'], # 强制指定颜色
        s=120, alpha=0.7,
        edgecolor='w', linewidth=0.5
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title='Domain Source', fontsize=11, title_fontsize=12, loc='upper right')
    
    # 去掉刻度值 (t-SNE 坐标值无物理意义)
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存: {filename}")

if __name__ == "__main__":
    run_domain_shift_analysis()