import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# =============== 核心修改：设置中文字体 ===============
import platform

system_name = platform.system()
if system_name == "Windows":
    # Windows 系统通常使用 SimHei (黑体) 或 Microsoft YaHei (微软雅黑)
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
elif system_name == "Darwin":
    # Mac 系统通常使用 Arial Unicode MS 或 Heiti TC
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
else:
    # Linux 系统可能需要自行安装字体，这里尝试常见的
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 
# ===================================================
# ================= 配置区域 =================
# 1. 你的数据集路径 (如果你已经改名了，就用新文件；没改名用旧文件)
DATASET_PATH = 'D:/research\Reconstruction-master\data\dataset2\prudataset.pkl' 
# DATASET_PATH = 'dataset2/dataset2_addedlabel.pkl' # 如果还没改名用这个

# 2. 选择交互模式 ('2', '3', '5')，注意你描述里还有 '0'，看你需要哪个
INTERACTION_MODE = '3'                

# 3. 选择一个用户进行分析 ('yzx' 或 'zwj')
USER_NAME = 'yzx'                     

# 4. 你想分析的物体列表 
# 注意：如果你已经运行了改名脚本，这里必须用英文名！
# 如果还没运行改名脚本，请把这里改成中文名 (如 '七喜', '圆柱体' 等)
TARGET_OBJECTS = [
    'SodaBtl', 'Cylinder', 'Cuboid', 'ChipTube', 
    'WineGlass', 'SportBtl', 'Triprism', 'Goblet','JuiceBtl','Cube',
    'WaterBtl','TeaBtl'
]

def load_and_extract_features(pkl_path, user, mode, target_objs=None):
    print(f"正在加载数据集: {pkl_path}...")
    if not os.path.exists(pkl_path):
        print(f"❌ 错误: 找不到文件 {pkl_path}")
        return None, None, None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if user not in data:
        print(f"❌ 错误: 用户 {user} 不在数据集中。可用用户: {list(data.keys())}")
        return None, None, None

    # 容器
    pressure_features = []
    joint_features = []
    labels = []

    user_data = data[user]
    
    # 确定要遍历的物体
    available_objs = list(user_data.keys())
    if target_objs is None:
        objs_to_process = available_objs
    else:
        # 只取数据集中存在的物体
        objs_to_process = [o for o in target_objs if o in available_objs]

    print(f"正在分析 {len(objs_to_process)} 个物体: {objs_to_process}")

    for obj_name in objs_to_process:
        # 检查该物体是否有对应的 Mode 数据
        if mode not in user_data[obj_name]:
            print(f"⚠️ 跳过 {obj_name}: 没有 mode '{mode}' 的数据")
            continue

        # 获取原始数据 
        # 结构: data[user][obj][mode]['pressure']
        # 假设形状是 [Batch_Size, Seq_Len, Dim]
        raw_pressure = user_data[obj_name][mode]['pressure'] 
        raw_joints = user_data[obj_name][mode]['joints']
        
        # === 特征预处理 (Flatten) ===
        # 我们需要把 (Batch, Time, Dim) 展平成 (Batch, Time*Dim) 才能喂给 t-SNE
        # 比如 Pressure: (100, 20, 16) -> (100, 320)
        
        # 确保数据是 numpy 数组
        if not isinstance(raw_pressure, np.ndarray):
            raw_pressure = np.array(raw_pressure)
        if not isinstance(raw_joints, np.ndarray):
            raw_joints = np.array(raw_joints)

        # 展平除 Batch (第0维) 以外的所有维度
        p_flat = raw_pressure.reshape(raw_pressure.shape[0], -1)
        j_flat = raw_joints.reshape(raw_joints.shape[0], -1)

        pressure_features.append(p_flat)
        joint_features.append(j_flat)
        
        # 记录标签 (每个样本都要一个标签)
        labels.extend([obj_name] * raw_pressure.shape[0])

    # 拼接所有物体的数据
    if len(pressure_features) == 0:
        print("❌ 未提取到任何数据，请检查配置。")
        return None, None, None

    X_pressure = np.concatenate(pressure_features, axis=0)
    X_joints = np.concatenate(joint_features, axis=0)
    y_labels = np.array(labels)

    return X_pressure, X_joints, y_labels

def plot_tsne(X, y, title, save_name):
    print(f"正在运行 t-SNE: {title} (输入形状: {X.shape})...")
    
    # 1. 数据标准化 (Standardization) - 对 t-SNE 很重要
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA 预降维 (如果特征维度太高 > 50，先用 PCA 降一下加速)
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)

    # 3. t-SNE 降维 (降到 2D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_scaled)

    # 4. 绘图
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=y, 
        palette=sns.color_palette("bright", len(np.unique(y))), # 使用鲜艳的颜色
        style=y, # 不同物体用不同形状的点，增加区分度
        legend='full',
        s=80, alpha=0.7
    )
    plt.title(f'{title} Cluster Analysis (t-SNE)', fontsize=25)
    plt.xlabel('Dimension 1', fontsize=25)
    plt.ylabel('Dimension 2', fontsize=25)
    plt.legend(
        loc='best',          # 'best' 会自动寻找没有数据点的空位，防止挡住点
        fontsize=14,         # 设置字体大小 (默认通常是 10，14 会大一圈)
        framealpha=0.7       # (可选) 让图例背景稍微透明一点，即使挡住点也能看到后面
    )
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_name, dpi=300)
    print(f"✅ 图片已保存: {save_name}")
    # plt.show() # 如果是远程服务器，请注释这行

def calculate_intra_class_variance(pkl_path, user, mode, target_objs):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n{'='*20} 物体内部差异分析 {'='*20}")
    print(f"{'物体名称':<10} | {'平均样本间距 (Norm)':<20} | {'数据稳定性'}")
    print("-" * 50)
    
    for obj in target_objs:
        if obj not in data[user]: continue
        
        # 获取原始数据 (Samples, Seq, Dim)
        raw = np.array(data[user][obj][mode]['pressure'])
        # 展平
        flat = raw.reshape(raw.shape[0], -1)
        
        # 计算所有样本到“平均样本”的距离
        mean_sample = np.mean(flat, axis=0)
        distances = np.linalg.norm(flat - mean_sample, axis=1)
        avg_dist = np.mean(distances)
        
        stability = "⭐⭐⭐⭐⭐ (极稳)" if avg_dist < 5 else \
                    "⭐⭐⭐ (一般)" if avg_dist < 15 else "⭐ (多变)"
        
        print(f"{obj:<10} | {avg_dist:<20.4f} | {stability}")
    print("=" * 50)

if __name__ == "__main__":
    # 1. 加载数据
    X_press, X_joint, y_labels = load_and_extract_features(
        DATASET_PATH, USER_NAME, INTERACTION_MODE, TARGET_OBJECTS
    )
    calculate_intra_class_variance(
        DATASET_PATH, USER_NAME, INTERACTION_MODE, TARGET_OBJECTS
    )
    if X_press is not None:
        print(f"共提取样本数: {len(y_labels)}")
        
        # 2. 绘制 压力 (Pressure) 聚类图
        plot_tsne(X_press, y_labels, 
                  title=f"Tactile Pressure Clustering", 
                  save_name="cluster_pressure.png")

        # 3. 绘制 关节 (Joints) 聚类图
        plot_tsne(X_joint, y_labels, 
                  title=f"Proprioception (Joints) Clustering", 
                  save_name="cluster_joints.png")
        
        print("\n全部完成！请查看生成的 .png 图片。")