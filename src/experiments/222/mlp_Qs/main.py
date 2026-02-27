# main.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入自定义模块
from data_processing import load_or_create_data, preprocess_data, get_data_info, save_data_info
from model import create_regression_model, save_model_params

def train_regression_model(model, X_train, y_train, criterion, optimizer, 
                           epochs=100, batch_size=32, val_data=None, verbose=True):
    """
    训练回归模型
    
    参数:
        model: 要训练的模型
        X_train: 训练特征
        y_train: 训练标签
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        batch_size: 批次大小
        val_data: 验证数据 (X_val, y_val)
        verbose: 是否打印训练过程
    
    返回:
        model: 训练后的模型
        history: 训练历史记录
    """
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_mae': [], 'val_r2': []}
    n_samples = X_train.shape[0]
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # 回归问题，需要reshape
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # 随机打乱数据
        indices = torch.randperm(n_samples)
        X_shuffled = X_train_tensor[indices]
        y_shuffled = y_train_tensor[indices]
        
        # 批次训练
        for i in range(0, n_samples, batch_size):
            # 获取批次
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / (n_samples / batch_size)
        history['train_loss'].append(avg_loss)
        
        # 验证（如果有验证数据）
        if val_data is not None:
            X_val, y_val = val_data
            val_loss, val_mse, val_mae, val_r2 = evaluate_regression_model(model, X_val, y_val, criterion)
            history['val_loss'].append(val_loss)
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
        
        # 打印训练信息
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}", end="")
            if val_data is not None:
                print(f", Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}")
            else:
                print()
    
    return model, history

def evaluate_regression_model(model, X_test, y_test, criterion=None):
    """
    评估回归模型
    
    参数:
        model: 要评估的模型
        X_test: 测试特征
        y_test: 测试标签
        criterion: 损失函数（可选）
    
    返回:
        loss, mse, mae, r2: 评估指标
    """
    model.eval()
    
    with torch.no_grad():
        # 转换为张量
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        
        # 预测
        outputs = model(X_test_tensor)
        predictions = outputs.numpy().flatten()
        y_true = y_test.flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        # 计算损失（如果提供了损失函数）
        if criterion is not None:
            loss = criterion(outputs, y_test_tensor).item()
        else:
            loss = None
    
    return loss, mse, mae, r2

def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    # 绘制训练损失
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证MSE和MAE
    if 'val_mse' in history and 'val_mae' in history:
        plt.subplot(1, 3, 2)
        plt.plot(history['val_mse'], label='Val MSE', color='green')
        plt.plot(history['val_mae'], label='Val MAE', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Validation Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 绘制验证R²
    if 'val_r2' in history:
        plt.subplot(1, 3, 3)
        plt.plot(history['val_r2'], label='Val R²', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.title('Validation R² Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_true(y_true, y_pred, title="Prediction vs Truth"):
    """绘制预测值与真实值的对比图"""
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.subplot(2, 1, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=1)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} (Scatter Plot)')
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 1, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{title} (Residual Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names=None):
    """绘制特征重要性（基于第一层权重）"""
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(model.input_dim)]
    
    # 获取第一层权重
    first_layer = model.network[0]
    if isinstance(first_layer, nn.Linear):
        weights = first_layer.weight.detach().numpy()
        
        # 计算每个特征的绝对权重和
        importance = np.mean(np.abs(weights), axis=0)
        
        # 归一化
        importance = importance / np.sum(importance)
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance (Based on First Layer Weights)')
        plt.tight_layout()
        plt.show()
    else:
        print("Unable to get feature importance: first layer is not a linear layer")
def main():
    """Main function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("MLP回归模型训练")
    print("="*60)
    
    # 1. 加载和预处理数据
    print("\n1. 加载和预处理数据...")
    try:
        X_train, X_test, y_train, y_test = load_or_create_data(
            data_path="../../data/data_222.npz", create_sample=False
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Data preprocessing (standardization)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler = preprocess_data(
        X_train, X_test, y_train, y_test
    )
    
    # 获取数据信息
    print("\n获取数据信息...")
    data_info = get_data_info(X_train_scaled, y_train_scaled)
    save_data_info(data_info, "../../args/data_info.npz")
    
    # 保存标准化器
    np.savez("../../args/scalers.npz", 
             X_scaler_mean=X_scaler.mean_,
             X_scaler_scale=X_scaler.scale_,
             y_scaler_mean=y_scaler.mean_,
             y_scaler_scale=y_scaler.scale_)
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    model = create_regression_model(
        input_dim=data_info['n_features'],
        model_config={
            'hidden_dims': [128, 64, 32],  # 增加网络深度以处理复杂函数
            'dropout_rate': 0.1,  # 降低dropout率
            'activation': 'tanh'  # 使用tanh激活函数可能更适合回归
        }
    )
    
    # 3. 定义损失函数和优化器
    print("\n3. 定义损失函数和优化器...")
    criterion = nn.MSELoss()  # 回归问题使用均方误差损失
    print("  使用均方误差损失 (MSELoss)")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化
    print(f"  使用Adam优化器，学习率: 0.001，权重衰减: 1e-5")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    # 对于大数据集，减少训练轮数并增加批次大小
    epochs = 50
    batch_size = 1024
    
    model, history = train_regression_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        val_data=(X_test_scaled, y_test_scaled),
        verbose=True
    )
    
    # 5. 评估模型
    print("\n5. 评估模型...")
    loss, mse, mae, r2 = evaluate_regression_model(model, X_test_scaled, y_test_scaled, criterion)
    print(f"  测试集损失 (MSE): {mse:.6f}")
    print(f"  测试集MAE: {mae:.6f}")
    print(f"  测试集R²分数: {r2:.6f}")
    
    # 获取预测值用于可视化
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_scaled = model(X_test_tensor).numpy().flatten()
        
        # 反标准化预测值
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # 6. 可视化结果
    print("\n6. 可视化结果...")
    # 绘制训练历史
    plot_training_history(history)
    
    # 绘制预测值 vs 真实值
    # 只取部分数据以避免内存问题
    sample_size = min(10000, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    plot_predictions_vs_true(y_true[indices], y_pred[indices], "Prediction vs Truth")
    
    # 绘制特征重要性
    feature_names = [f'Feature {i+1}' for i in range(data_info['n_features'])]
    plot_feature_importance(model, feature_names)
    
    # 7. 保存模型参数
    print("\n7. 保存模型参数...")
    os.makedirs("../../args", exist_ok=True)
    save_model_params(model, "../../args", "args")
    
    # 8. 保存训练历史
    np.savez("../../args/training_history.npz", **history)
    print("训练历史已保存到 ../../args/training_history.npz")
    
    # 9. 保存预测结果示例
    example_indices = np.random.choice(len(y_true), 100, replace=False)
    example_results = {
        'true_values': y_true[example_indices],
        'predicted_values': y_pred[example_indices],
        'indices': example_indices
    }
    np.savez("../../args/example_predictions.npz", **example_results)
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"模型参数已保存到 ../../args/ 文件夹")
    print("="*60)

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("../../data", exist_ok=True)
    os.makedirs("../../args", exist_ok=True)
    
    # 运行主程序
    main()