# predict.py
import numpy as np
import torch
import os
from model import MLPRegressor

def load_model_and_scalers(model_path="../../args/args.pth", scaler_path="../../args/scalers.npz"):
    """加载模型和标准化器"""
    # 加载检查点
    checkpoint = torch.load(model_path)
    
    # 创建模型
    model = MLPRegressor(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_dims=checkpoint['model_config']['hidden_dims'],
        dropout_rate=0.0,  # 预测时关闭dropout
        activation='tanh'
    )
    
    # 加载参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载标准化器参数
    scaler_data = np.load(scaler_path)
    X_scaler = {'mean': scaler_data['X_scaler_mean'], 'scale': scaler_data['X_scaler_scale']}
    y_scaler = {'mean': scaler_data['y_scaler_mean'], 'scale': scaler_data['y_scaler_scale']}
    
    return model, X_scaler, y_scaler

def predict(model, X, X_scaler, y_scaler):
    """使用模型进行预测"""
    # 标准化输入
    X_scaled = (X - X_scaler['mean']) / X_scaler['scale']
    
    # 预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_pred_scaled = model(X_tensor).numpy().flatten()
    
    # 反标准化输出
    y_pred = y_pred_scaled * y_scaler['scale'] + y_scaler['mean']
    
    return y_pred

def main():
    """主函数"""
    print("加载模型...")
    model, X_scaler, y_scaler = load_model_and_scalers()
    print("模型加载完成")
    
    # 示例：创建一个8维超球面上的随机点
    n_samples = 10
    # 生成随机向量
    random_vectors = np.random.randn(n_samples, 8)
    # 归一化到单位超球面
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    points_on_sphere = random_vectors / norms
    
    print(f"\n生成 {n_samples} 个超球面上的点:")
    for i, point in enumerate(points_on_sphere):
        print(f"点 {i+1}: {point}")
    
    # 预测Q值
    q_values = predict(model, points_on_sphere, X_scaler, y_scaler)
    
    print(f"\n预测的Q值:")
    for i, q in enumerate(q_values):
        print(f"点 {i+1}: Q(s) = {q:.6f}")
    
    # 保存预测结果
    np.savez("../../args/predictions_example.npz", 
             points=points_on_sphere, 
             q_values=q_values)
    
    print(f"\n预测结果已保存到 ../../args/predictions_example.npz")

if __name__ == "__main__":
    main()