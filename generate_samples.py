#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成拉丁超立方采样点用于优化

该脚本将生成200-300组参数组合，用于CFD计算
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import os

def generate_parameter_samples(n_samples=250):
    """
    使用拉丁超立方采样生成参数组合
    
    参数范围（需要根据您的具体需求调整）:
    Tt: [0.5, 1.5]
    Ts: [0.5, 1.5]
    Tad: [0.1, 1.0]
    Tb: [0.1, 1.0]
    """
    
    # 定义参数范围
    # 每一行是一个参数的 [最小值, 最大值]
    bounds = np.array([
        [0.5, 1.5],  # Tt
        [0.5, 1.5],  # Ts
        [0.1, 1.0],  # Tad
        [0.1, 1.0]   # Tb
    ])
    
    # 创建拉丁超立方采样器
    sampler = qmc.LatinHypercube(d=4, seed=42)  # 4个参数，固定随机种子以确保可重复性
    
    # 生成采样点
    sample = sampler.random(n=n_samples)
    
    # 将采样点缩放到实际参数范围
    sample_scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    # 创建DataFrame
    df = pd.DataFrame(sample_scaled, columns=['Tt', 'Ts', 'Tad', 'Tb'])
    
    # 添加案例编号
    df.insert(0, 'case_id', range(1, len(df) + 1))
    
    return df

def main():
    # 生成250个采样点
    n_samples = 250
    print(f"生成 {n_samples} 个采样点...")
    
    df = generate_parameter_samples(n_samples)
    
    # 保存到CSV文件
    output_file = os.path.join("csv_data", "parameter_samples.csv")
    df.to_csv(output_file, index=False)
    print(f"采样点已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n参数范围统计:")
    print(df.describe())
    
    # 显示前几行
    print("\n前5行数据:")
    print(df.head())
    
    print(f"\n总共生成了 {len(df)} 个参数组合")
    print("现在可以使用 run_all_cases.py 来运行所有案例")

if __name__ == "__main__":
    main()