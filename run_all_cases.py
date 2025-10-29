#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量运行所有优化案例

该脚本将:
1. 读取 csv_data/parameter_samples.csv 文件
2. 对每个案例调用 run_optimization_case.py
3. 收集结果并保存到 csv_data/final_results.csv
"""

import pandas as pd
import subprocess
import sys
import os
import time

def run_single_case(case_id, Tt, Ts, Tad, Tb):
    """
    运行单个案例
    """
    print(f"\n运行案例 {case_id}: Tt={Tt:.3f}, Ts={Ts:.3f}, Tad={Tad:.3f}, Tb={Tb:.3f}")
    
    # 调用单个案例运行脚本
    cmd = [
        "python", "run_optimization_case.py",
        str(Tt), str(Ts), str(Tad), str(Tb),
        "--output", "csv_data/temp_result.csv"
    ]
    
    try:
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)  # 1小时超时
        print(f"案例 {case_id} 执行成功")
        
        # 读取结果
        if os.path.exists("csv_data/temp_result.csv") and os.path.getsize("csv_data/temp_result.csv") > 0:
            with open("csv_data/temp_result.csv", "r") as f:
                result_line = f.read().strip()
            os.remove("csv_data/temp_result.csv")  # 清理临时文件
            return result_line
        else:
            print(f"案例 {case_id} 没有产生结果")
            return f"{case_id},{Tt},{Ts},{Tad},{Tb},0,0,0"
            
    except subprocess.CalledProcessError as e:
        print(f"案例 {case_id} 执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return f"{case_id},{Tt},{Ts},{Tad},{Tb},0,0,0"
    except subprocess.TimeoutExpired:
        print(f"案例 {case_id} 执行超时")
        return f"{case_id},{Tt},{Ts},{Tad},{Tb},0,0,0"

def main():
    # 检查参数采样文件是否存在
    sample_file = os.path.join("csv_data", "parameter_samples.csv")
    if not os.path.exists(sample_file):
        print(f"错误: 找不到参数采样文件 '{sample_file}'")
        print("请先运行 generate_samples.py 生成采样点")
        return 1
    
    # 读取参数采样文件
    print("读取参数采样文件...")
    df = pd.read_csv(sample_file)
    print(f"总共需要运行 {len(df)} 个案例")
    
    # 准备结果文件
    result_file = os.path.join("csv_data", "final_results.csv")
    result_header = "case_id,Tt,Ts,Tad,Tb,Nu,f,target_param\n"
    
    # 写入头部
    with open(result_file, 'w') as f:
        f.write(result_header)
    
    # 逐个运行案例
    start_time = time.time()
    for index, row in df.iterrows():
        case_id = row['case_id']
        Tt = row['Tt']
        Ts = row['Ts']
        Tad = row['Tad']
        Tb = row['Tb']
        
        # 运行案例
        result_line = run_single_case(case_id, Tt, Ts, Tad, Tb)
        
        # 保存结果
        with open(result_file, 'a') as f:
            f.write(result_line + '\n')
        
        # 显示进度
        elapsed_time = time.time() - start_time
        avg_time_per_case = elapsed_time / (index + 1)
        remaining_cases = len(df) - (index + 1)
        estimated_remaining_time = avg_time_per_case * remaining_cases
        
        print(f"进度: {index + 1}/{len(df)} ({(index + 1)/len(df)*100:.1f}%)")
        print(f"预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
    
    total_time = time.time() - start_time
    print(f"\n所有案例运行完成!")
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"结果已保存到: {result_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())