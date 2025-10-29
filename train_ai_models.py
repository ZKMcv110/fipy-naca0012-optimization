#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练AI模型用于预测Nu和f

该脚本将:
1. 读取 csv_data/final_results.csv 文件
2. 使用PyTorch训练两个神经网络模型（Nu和f）
3. 保存训练好的模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# 定义神经网络模型
class ParameterToPerformanceNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super(ParameterToPerformanceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data():
    """加载并预处理数据"""
    # 检查结果文件是否存在
    result_file = os.path.join("csv_data", "final_results.csv")
    if not os.path.exists(result_file):
        print(f"错误: 找不到结果文件 '{result_file}'")
        print("请先运行 run_all_cases.py 生成结果数据")
        return None, None, None, None, None, None
    
    # 读取数据
    df = pd.read_csv(result_file)
    print(f"加载了 {len(df)} 个数据点")
    
    # 移除包含0值的行（失败的案例）
    df = df[(df['Nu'] != 0) & (df['f'] != 0)]
    print(f"移除失败案例后剩余 {len(df)} 个数据点")
    
    # 提取输入和输出
    X = df[['Tt', 'Ts', 'Tad', 'Tb']].values
    y_Nu = df['Nu'].values.reshape(-1, 1)
    y_f = df['f'].values.reshape(-1, 1)
    
    # 标准化输入数据
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 标准化输出数据
    scaler_Nu = StandardScaler()
    y_Nu_scaled = scaler_Nu.fit_transform(y_Nu)
    
    scaler_f = StandardScaler()
    y_f_scaled = scaler_f.fit_transform(y_f)
    
    return X_scaled, y_Nu_scaled, y_f_scaled, scaler_X, scaler_Nu, scaler_f

def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.001):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # 保存最佳模型
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
    
    # 恢复最佳模型状态
    model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def main():
    print("加载并预处理数据...")
    X_scaled, y_Nu_scaled, y_f_scaled, scaler_X, scaler_Nu, scaler_f = load_and_preprocess_data()
    
    if X_scaled is None:
        return 1
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_scaled)
    y_Nu_tensor = torch.FloatTensor(y_Nu_scaled)
    y_f_tensor = torch.FloatTensor(y_f_scaled)
    
    # 划分训练集和验证集
    X_train, X_val, y_Nu_train, y_Nu_val, y_f_train, y_f_val = train_test_split(
        X_tensor, y_Nu_tensor, y_f_tensor, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    
    # 训练Nu预测模型
    print("\n训练Nu预测模型...")
    model_Nu = ParameterToPerformanceNet(input_size=4, hidden_size=64, output_size=1)
    train_losses_Nu, val_losses_Nu = train_model(
        model_Nu, X_train, y_Nu_train, X_val, y_Nu_val, epochs=1000, lr=0.001
    )
    
    # 训练f预测模型
    print("\n训练f预测模型...")
    model_f = ParameterToPerformanceNet(input_size=4, hidden_size=64, output_size=1)
    train_losses_f, val_losses_f = train_model(
        model_f, X_train, y_f_train, X_val, y_f_val, epochs=1000, lr=0.001
    )
    
    # 保存模型和标准化器
    print("\n保存模型和标准化器...")
    torch.save(model_Nu.state_dict(), 'model_Nu.pth')
    torch.save(model_f.state_dict(), 'model_f.pth')
    torch.save(scaler_X, 'scaler_X.pkl')
    torch.save(scaler_Nu, 'scaler_Nu.pkl')
    torch.save(scaler_f, 'scaler_f.pkl')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_Nu, label='Training Loss')
    plt.plot(val_losses_Nu, label='Validation Loss')
    plt.title('Nu Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses_f, label='Training Loss')
    plt.plot(val_losses_f, label='Validation Loss')
    plt.title('f Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    print("模型训练完成!")
    print("模型已保存: model_Nu.pth, model_f.pth")
    print("标准化器已保存: scaler_X.pkl, scaler_Nu.pkl, scaler_f.pkl")
    print("训练曲线已保存: training_curves.png")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())