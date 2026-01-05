# data_processing.py
import numpy as np
import os
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_or_create_data(data_path="../../data/data_222.npz", create_sample=False):
    """
    加载或创建数据
    
    参数:
        data_path: 数据文件路径
        create_sample: 如果数据不存在，是否创建示例数据
    
    返回:
        X_train, X_test, y_train, y_test: 训练集和测试集
    """
    if os.path.exists(data_path):
        # 加载现有数据
        print(f"从 {data_path} 加载数据...")
        data = np.load(data_path)
        
        # 获取文件中的键
        keys = list(data.keys())
        print(f"数据文件包含的键: {keys}")
        
        # 根据不同的数据格式进行处理
        if 'inputs' in keys and 'labels' in keys:
            # 格式：有'inputs'和'labels'
            X = data['inputs']
            y = data['labels']
            
            # 只取标签的第一维作为目标值
            if len(y.shape) == 2 and y.shape[1] == 2:
                print("检测到2维标签，只取第一维作为目标值")
                y = y[:, 0]
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        elif 'X_train' in keys and 'y_train' in keys:
            # 格式1: 已经划分好的训练集和测试集
            X_train = data['X_train']
            y_train = data['y_train']
            
            if 'X_test' in keys and 'y_test' in keys:
                X_test = data['X_test']
                y_test = data['y_test']
            else:
                # 如果没有测试集，使用训练集作为测试集
                X_test, y_test = X_train, y_train
                
        elif 'X' in keys and 'y' in keys:
            # 格式2: 有明确的X和y
            X = data['X']
            y = data['y']
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            # 格式3: 其他格式，使用前两个数组
            if len(keys) >= 2:
                X = data[keys[0]]
                y = data[keys[1]]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                raise ValueError(f"数据文件格式不支持。请确保文件包含至少2个数组。")
                
    elif create_sample:
        # 创建示例数据
        print(f"文件 {data_path} 不存在，创建示例数据...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # 创建回归数据
        X, y = make_regression(
            n_samples=100000, n_features=8, noise=0.1, random_state=42
        )
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建2维标签（只使用第一维）
        y_2d = np.zeros((len(y), 2))
        y_2d[:, 0] = y
        
        # 保存数据
        np.savez(data_path, 
                 inputs=X, labels=y_2d)
    else:
        raise FileNotFoundError(f"数据文件 {data_path} 不存在且未启用创建示例数据模式")
    
    print(f"训练集: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    print(f"测试集: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test, y_train=None, y_test=None):
    """
    数据预处理（标准化）
    
    参数:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
    
    返回:
        标准化后的数据和标准化器
    """
    # 标准化特征
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # 标准化标签（可选）
    if y_train is not None and y_test is not None:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"特征标准化完成。")
        print(f"标签标准化完成。标签均值: {y_scaler.mean_[0]:.4f}, 标准差: {y_scaler.scale_[0]:.4f}")
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler
    else:
        print(f"特征标准化完成。")
        return X_train_scaled, X_test_scaled, X_scaler

def get_data_info(X_train, y_train):
    """
    获取数据信息
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
    
    返回:
        info_dict: 包含数据信息的字典
    """
    info = {
        'n_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'y_mean': np.mean(y_train),
        'y_std': np.std(y_train),
        'y_min': np.min(y_train),
        'y_max': np.max(y_train),
        'y_median': np.median(y_train)
    }
    
    print("数据信息:")
    print(f"  样本数: {info['n_samples']}")
    print(f"  特征数: {info['n_features']}")
    print(f"  目标值统计:")
    print(f"    均值: {info['y_mean']:.4f}")
    print(f"    标准差: {info['y_std']:.4f}")
    print(f"    最小值: {info['y_min']:.4f}")
    print(f"    最大值: {info['y_max']:.4f}")
    print(f"    中位数: {info['y_median']:.4f}")
    
    return info

def save_data_info(info, save_path="../../args/data_info.npz"):
    """
    保存数据信息
    
    参数:
        info: 数据信息字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **info)
    print(f"数据信息已保存到 {save_path}")