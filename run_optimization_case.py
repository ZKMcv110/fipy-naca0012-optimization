#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化案例运行脚本
此脚本将:
1. 调用naca0012dat.py生成几何文件 (需要从命令行参数获取参数)
2. 生成网格
3. 运行求解器计算Nu和f
4. 输出结果
"""

import subprocess
import sys
import os
import argparse

def run_naca0012_generator(Tt, Ts, Ta, Tad, Twa, Tb):
    """
    调用naca0012dat.py生成几何文件
    """
    try:
        # 调用naca0012dat.py生成几何文件
        cmd = [
            "python", "naca0012dat.py",
            "--Tt", str(Tt),
            "--Ts", str(Ts),
            "--Ta", str(Ta),
            "--Tad", str(Tad),
            "--Twa", str(Twa),
            "--Tb", str(Tb)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("naca0012dat.py执行成功")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"naca0012dat.py执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("未找到naca0012dat.py脚本")
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
        cmd = ["python", "求解器_pvtnf.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("求解器执行成功")
        print(result.stdout)
        
        # 从保存的结果文件中读取Nu和f
        solver_results_file = os.path.join('solver_results', 'latest_solution.pkl')
        if os.path.exists(solver_results_file):
            import pickle
            with open(solver_results_file, 'rb') as f:
                results = pickle.load(f)
                Nu = results.get('Nu', 0)
                f_val = results.get('f', 0)
                return Nu, f_val
        else:
            print("未找到求解器结果文件")
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
    parser.add_argument('Tt', type=float, help='Tt 参数 (横向间距系数)')
    parser.add_argument('Ts', type=float, help='Ts 参数 (纵向间距系数)')
    parser.add_argument('Tad', type=float, help='Tad 参数 (交错间距系数)')
    parser.add_argument('Tb', type=float, help='Tb 参数 (厚度系数)')
    parser.add_argument('--Ta', type=float, default=0.0, help='Ta 参数 (弯度系数，默认为0.0)')
    parser.add_argument('--Twa', type=float, default=0.5, help='Twa 参数 (宽长比，默认为0.5)')
    parser.add_argument('--output', type=str, default=os.path.join('csv_data', 'result.csv'), help='结果输出文件')
    
    args = parser.parse_args()
    
    print(f"运行优化案例: Tt={args.Tt}, Ts={args.Ts}, Ta={args.Ta}, Tad={args.Tad}, Twa={args.Twa}, Tb={args.Tb}")
    
    # 1. 调用naca0012dat.py生成几何文件
    print("\n1. 生成几何文件...")
    if not run_naca0012_generator(args.Tt, args.Ts, args.Ta, args.Tad, args.Twa, args.Tb):
        print("几何文件生成失败，终止流程")
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
        f.write(f"{args.Tt},{args.Ts},{args.Ta},{args.Tad},{args.Twa},{args.Tb},{Nu},{f},{target_param}\n")
    
    print(f"结果已写入: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())