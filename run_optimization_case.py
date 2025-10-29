#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化案例运行脚本
此脚本将:
1. 调用C++程序生成几何文件 (需要从命令行参数获取参数)
2. 生成网格
3. 运行求解器计算Nu和f
4. 输出结果
"""

import subprocess
import sys
import os
import argparse

def run_cplusplus_geometry_engine(Tt, Ts, Tad, Tb):
    """
    调用C++几何引擎生成几何文件
    """
    try:
        # 假设C++程序名为 geometry_engine.exe，接受4个参数
        # 您需要根据实际的C++程序接口修改此命令
        cmd = ["geometry_engine.exe", str(Tt), str(Ts), str(Tad), str(Tb)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("C++几何引擎执行成功")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"C++几何引擎执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("未找到C++几何引擎程序 (geometry_engine.exe)")
        return False

def run_mesh_generator():
    """
    运行网格生成器
    """
    try:
        # 删除已存在的网格文件以强制重新生成
        mesh_file = "airfoil_array.msh2"
        if os.path.exists(mesh_file):
            os.remove(mesh_file)
            print(f"已删除旧网格文件: {mesh_file}")
        
        # 运行网格生成器
        cmd = ["python", "网格生成器2_三角形.py", "-nopopup"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("网格生成器执行成功")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"网格生成器执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("未找到网格生成器脚本")
        return False

def run_solver():
    """
    运行求解器计算Nu和f
    """
    try:
        # 运行求解器
        cmd = ["python", "compute_Nu_f.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("求解器执行成功")
        print(result.stdout)
        
        # 解析输出获取Nu和f
        lines = result.stdout.strip().split('\n')
        last_line = lines[-1]
        if ',' in last_line:
            Nu, f = map(float, last_line.split(','))
            return Nu, f
        else:
            print("无法解析求解器输出")
            return 0, 0
    except subprocess.CalledProcessError as e:
        print(f"求解器执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return 0, 0
    except FileNotFoundError:
        print("未找到求解器脚本")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description='运行优化案例')
    parser.add_argument('Tt', type=float, help='Tt 参数')
    parser.add_argument('Ts', type=float, help='Ts 参数')
    parser.add_argument('Tad', type=float, help='Tad 参数')
    parser.add_argument('Tb', type=float, help='Tb 参数')
    parser.add_argument('--output', type=str, default=os.path.join('csv_data', 'result.csv'), help='结果输出文件')
    
    args = parser.parse_args()
    
    print(f"运行优化案例: Tt={args.Tt}, Ts={args.Ts}, Tad={args.Tad}, Tb={args.Tb}")
    
    # 1. 调用C++几何引擎
    print("\n1. 调用C++几何引擎...")
    if not run_cplusplus_geometry_engine(args.Tt, args.Ts, args.Tad, args.Tb):
        print("C++几何引擎执行失败，终止流程")
        return 1
    
    # 2. 生成网格
    print("\n2. 生成网格...")
    if not run_mesh_generator():
        print("网格生成失败，终止流程")
        return 1
    
    # 3. 运行求解器
    print("\n3. 运行求解器...")
    Nu, f = run_solver()
    
    # 4. 输出结果
    print(f"\n4. 输出结果...")
    print(f"Nu: {Nu}")
    print(f"f: {f}")
    if f > 0:
        target_param = Nu / (f**(1/3))
        print(f"目标参数 Nu/(f^(1/3)): {target_param}")
    else:
        target_param = 0
        print("无法计算目标参数")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入结果文件
    with open(args.output, 'a') as f:
        f.write(f"{args.Tt},{args.Ts},{args.Tad},{args.Tb},{Nu},{f},{target_param}\n")
    
    print(f"结果已写入: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())