# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegressor(nn.Module):
    """
    多层感知机（MLP）回归模型
    
    参数:
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        dropout_rate: Dropout比例
        activation: 激活函数
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2, activation='relu'):
        super(MLPRegressor, self).__init__()
        
        self.input_dim = input_dim
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层：回归问题，一个输出神经元
        layers.append(nn.Linear(prev_dim, 1))
        
        # 组合所有层
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)
    
    def predict(self, x):
        """预测"""
        with torch.no_grad():
            return self.forward(x)

def create_regression_model(input_dim, model_config=None):
    """
    创建回归模型
    
    参数:
        input_dim: 输入特征维度
        model_config: 模型配置字典
    
    返回:
        model: 创建的模型
    """
    if model_config is None:
        model_config = {
            'hidden_dims': [64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu'
        }
    
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        activation=model_config['activation']
    )
    
    print(f"创建MLP回归模型:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层: {model_config['hidden_dims']}")
    print(f"  输出维度: 1 (回归问题)")
    print(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def save_model_params(model, folder_path="../../args", filename="args"):
    """
    保存模型参数
    
    参数:
        model: 要保存的模型
        folder_path: 保存文件夹路径
        filename: 保存文件名（不含扩展名）
    """
    import os
    
    os.makedirs(folder_path, exist_ok=True)
    
    # 保存为PyTorch格式
    torch_path = os.path.join(folder_path, f"{filename}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dims': [layer.out_features for layer in model.network 
                           if isinstance(layer, nn.Linear)][:-1],
            'model_class': 'MLPRegressor'
        }
    }, torch_path)
    
    # 保存为NumPy格式
    npz_path = os.path.join(folder_path, f"{filename}.npz")
    import numpy as np
    model_params = {}
    for name, param in model.named_parameters():
        model_params[name] = param.detach().numpy()
    np.savez(npz_path, **model_params)
    
    print(f"模型参数已保存:")
    print(f"  PyTorch格式: {torch_path}")
    print(f"  NumPy格式: {npz_path}")
    
    return torch_path, npz_path

def load_model_params(model, file_path):
    """
    加载模型参数
    
    参数:
        model: 要加载参数的模型
        file_path: 参数文件路径
    
    返回:
        model: 加载参数后的模型
    """
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"从 {file_path} 加载模型参数")
    return model