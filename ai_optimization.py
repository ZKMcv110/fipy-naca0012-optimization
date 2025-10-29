#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用训练好的AI模型进行优化

该脚本将:
1. 加载训练好的AI模型
2. 使用差分进化算法寻找最优参数组合
3. 输出最优解
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import differential_evolution
import pickle
import os

# 定义神经网络模型（与训练时相同）
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

def load_models():
    """加载训练好的模型和标准化器"""
    if not os.path.exists('model_Nu.pth') or not os.path.exists('model_f.pth'):
        print("错误: 找不到训练好的模型文件")
        print("请先运行 train_ai_models.py 训练模型")
        return None, None, None, None, None
    
    # 加载标准化器
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    
    with open('scaler_Nu.pkl', 'rb') as f:
        scaler_Nu = pickle.load(f)
    
    with open('scaler_f.pkl', 'rb') as f:
        scaler_f = pickle.load(f)
    
    # 创建模型
    model_Nu = ParameterToPerformanceNet(input_size=4, hidden_size=64, output_size=1)
    model_f = ParameterToPerformanceNet(input_size=4, hidden_size=64, output_size=1)
    
    # 加载模型权重
    model_Nu.load_state_dict(torch.load('model_Nu.pth'))
    model_f.load_state_dict(torch.load('model_f.pth'))
    
    # 设置为评估模式
    model_Nu.eval()
    model_f.eval()
    
    return model_Nu, model_f, scaler_X, scaler_Nu, scaler_f

def objective_function(params, model_Nu, model_f, scaler_X, scaler_Nu, scaler_f):
    """
    优化目标函数
    最大化 Nu / (f^(1/3))
    """
    # 参数范围检查
    Tt, Ts, Tad, Tb = params
    
    # 标准化输入
    params_scaled = scaler_X.transform([params])
    params_tensor = torch.FloatTensor(params_scaled)
    
    # 使用模型预测
    with torch.no_grad():
        Nu_scaled = model_Nu(params_tensor)
        f_scaled = model_f(params_tensor)
        
        # 反标准化
        Nu = scaler_Nu.inverse_transform(Nu_scaled.numpy())[0, 0]
        f = scaler_f.inverse_transform(f_scaled.numpy())[0, 0]
    
    # 计算目标函数（注意我们是最小化，所以要加负号）
    if f > 0:
        target = Nu / (f**(1/3))
    else:
        target = 0  # 如果f<=0，则目标值为0
    
    # 返回负值因为我们使用的是最小化算法
    return -target

def main():
    print("加载AI模型...")
    model_Nu, model_f, scaler_X, scaler_Nu, scaler_f = load_models()
    
    if model_Nu is None:
        return 1
    
    print("AI模型加载成功!")
    
    # 定义参数边界（与采样时相同）
    bounds = [
        (0.5, 1.5),  # Tt
        (0.5, 1.5),  # Ts
        (0.1, 1.0),  # Tad
        (0.1, 1.0)   # Tb
    ]
    
    print("\n开始优化...")
    print("使用差分进化算法寻找最优参数组合...")
    
    # 运行差分进化优化
    result = differential_evolution(
        objective_function,
        bounds,
        args=(model_Nu, model_f, scaler_X, scaler_Nu, scaler_f),
        seed=42,
        maxiter=100,
        popsize=15,
        atol=1e-6,
        tol=1e-6
    )
    
    # 输出结果
    print("\n优化完成!")
    print(f"最优参数:")
    print(f"  Tt: {result.x[0]:.6f}")
    print(f"  Ts: {result.x[1]:.6f}")
    print(f"  Tad: {result.x[2]:.6f}")
    print(f"  Tb: {result.x[3]:.6f}")
    print(f"最优目标值 Nu/(f^(1/3)): {-result.fun:.6f}")
    
    # 保存最优参数
    with open(os.path.join('csv_data', 'optimal_parameters.txt'), 'w') as f:
        f.write(f"Tt: {result.x[0]:.6f}\n")
        f.write(f"Ts: {result.x[1]:.6f}\n")
        f.write(f"Tad: {result.x[2]:.6f}\n")
        f.write(f"Tb: {result.x[3]:.6f}\n")
        f.write(f"Target value Nu/(f^(1/3)): {-result.fun:.6f}\n")
    
    print("\n最优参数已保存到: csv_data/optimal_parameters.txt")
    
    # 使用AI模型预测最优参数的Nu和f值
    params_scaled = scaler_X.transform([result.x])
    params_tensor = torch.FloatTensor(params_scaled)
    
    with torch.no_grad():
        Nu_scaled = model_Nu(params_tensor)
        f_scaled = model_f(params_tensor)
        
        Nu = scaler_Nu.inverse_transform(Nu_scaled.numpy())[0, 0]
        f = scaler_f.inverse_transform(f_scaled.numpy())[0, 0]
    
    print(f"\nAI模型预测结果:")
    print(f"  Nu: {Nu:.6f}")
    print(f"  f: {f:.6f}")
    print(f"  Nu/(f^(1/3)): {Nu/(f**(1/3)):.6f}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())