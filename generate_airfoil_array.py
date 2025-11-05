#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据给定参数生成NACA翼型阵列的脚本

该脚本可以通过命令行参数设置翼型的各项参数，无需手动修改代码
"""

import numpy as np
import argparse


class ThetaParameters:
    """定义参数结构体"""
    def __init__(self, Tt=0.75, Ts=1.0, Ta=0.0, Tad=0.5, Twa=0.5, Tb=0.12):
        self.Tt = Tt  # 横向间距系数
        self.Ts = Ts  # 纵向间距系数
        self.Ta = Ta  # 弯度系数
        self.Tad = Tad  # 交错间距系数
        self.Twa = Twa  # 宽长比
        self.Tb = Tb  # 厚度系数


def generate_and_write_airfoil(fp, x_center, y_center, c, m, p, t):
    """
    生成单个NACA翼型并写入文件
    
    Args:
        fp: 文件指针
        x_center: 翼型中心X坐标
        y_center: 翼型中心Y坐标
        c: 弦长
        m: 最大弯度
        p: 最大弯度位置
        t: 最大厚度
    """
    N = 201  # 每个翼型的点数
    
    x = np.linspace(0, c, N)
    x_c = x / c
    
    # 计算厚度分布
    yt = (t / 0.2) * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2 + 
                      0.2843 * x_c**3 - 0.1015 * x_c**4)
    yt *= c
    
    # 计算弯度线和斜率
    yc = np.zeros(N)
    dyc_dx = np.zeros(N)
    
    if m > 0:
        for i in range(N):
            if x_c[i] < p:
                yc[i] = (m / (p**2)) * (2 * p * x_c[i] - x_c[i]**2)
                dyc_dx[i] = (m / (p**2)) * (2 * p - 2 * x_c[i])
            else:
                yc[i] = (m / ((1-p)**2)) * ((1-2*p) + 2*p*x_c[i] - x_c[i]**2)
                dyc_dx[i] = (m / ((1-p)**2)) * (2*p - 2*x_c[i])
        yc *= c
        dyc_dx /= c
    
    # 计算上下翼面坐标
    theta = np.arctan(dyc_dx)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    xu = x - yt * sin_theta
    yu = yc + yt * cos_theta
    xl = x + yt * sin_theta
    yl = yc - yt * cos_theta
    
    # 强制后缘闭合
    yu[-1] = yl[-1] = 0.0
    xu[-1] = xl[-1] = c
    
    # 写入上表面坐标
    for i in range(N):
        fp.write(f"{xu[i] + x_center:.8f}\t{yu[i] + y_center:.8f}\n")
    
    # 倒序写入下表面坐标
    for i in range(N-1, -1, -1):
        fp.write(f"{xl[i] + x_center:.8f}\t{yl[i] + y_center:.8f}\n")
    
    fp.write("\n\n")


def generate_airfoil_array(params, output_filename="airfoil_array_SYMMETRIC.dat"):
    """
    生成翼型阵列并保存到文件
    
    Args:
        params: ThetaParameters 对象，包含所有参数
        output_filename: 输出文件名
    """
    # 转换为物理尺寸
    La = 1.0
    c = La
    m = params.Ta * c
    p = 0.0
    t = params.Tb * c
    Ls = params.Ts * c
    Lt = params.Tt * c
    Lad = params.Tad * c
    L1 = Ls + La
    
    num_rows = 3
    num_cols = 8
    
    with open(output_filename, "w") as fp:
        print(f"正在生成 {num_rows}x{num_cols} 对称翼型扰流柱阵列...")
        print(f"参数: Ta={params.Ta}, Tb={params.Tb}, Tt={params.Tt}, Ts={params.Ts}, Tad={params.Tad}, Twa={params.Twa}")
        
        for j in range(num_rows):
            for i in range(num_cols):
                x_center = i * L1 + (j % 2 != 0) * Lad
                y_center = (j - (num_rows - 1.0) / 2.0) * Lt
                
                generate_and_write_airfoil(fp, x_center, y_center, c, m, p, t)
    
    print(f"阵列数据已成功输出到 {output_filename}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成NACA翼型阵列')
    parser.add_argument('--Tt', type=float, default=0.75, help='横向间距系数')
    parser.add_argument('--Ts', type=float, default=1.0, help='纵向间距系数')
    parser.add_argument('--Ta', type=float, default=0.0, help='弯度系数')
    parser.add_argument('--Tad', type=float, default=0.5, help='交错间距系数')
    parser.add_argument('--Twa', type=float, default=0.5, help='宽长比')
    parser.add_argument('--Tb', type=float, default=0.12, help='厚度系数')
    parser.add_argument('--output', type=str, default="airfoil_array_SYMMETRIC.dat", help='输出文件名')
    
    args = parser.parse_args()
    
    # 定义参数
    params = ThetaParameters(args.Tt, args.Ts, args.Ta, args.Tad, args.Twa, args.Tb)
    
    # 生成翼型阵列
    generate_airfoil_array(params, args.output)


if __name__ == "__main__":
    main()