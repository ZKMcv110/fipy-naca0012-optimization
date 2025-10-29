# NACA0012翼型扰流柱优化项目

本项目旨在通过计算流体力学(CFD)和人工智能(AI)技术，寻找最优的扰流柱结构参数，以最大化换热效率与流动阻力的比值(Nu/(f^(1/3)))。

## 项目概述

本项目基于FiPy（一个用Python编写的偏微分方程求解器）开发，用于模拟NACA0012翼型周围的流体流动和传热过程。通过结合AI优化算法，自动寻找最优的扰流柱几何参数。

## 项目结构

```
fipy_naca0012_2.0/
├── 网格生成器2_三角形.py      # 网格生成脚本
├── 求解器2.py               # 基础CFD求解器
├── 求解器_pvtnf.py          # 带传热计算的CFD求解器
├── compute_Nu_f.py          # Nu和f参数计算脚本
├── post_processing.py       # 基础后处理脚本
├── enhanced_post_processing.py # 增强后处理脚本
├── run_optimization_case.py # 单个优化案例运行脚本
├── run_all_cases.py         # 批量运行所有案例脚本
├── generate_samples.py      # 生成参数采样点脚本
├── train_ai_models.py       # 训练AI模型脚本
├── ai_optimization.py       # AI优化脚本
├── requirements.txt         # 项目依赖包列表
├── README.md               # 项目说明书
├── airfoil_array_SYMMETRIC.dat  # 翼型坐标数据文件
├── airfoil_array.msh2       # 网格文件
├── gmsh.exe                 # 网格生成工具
├── gmsh-4.13.dll            # 网格生成工具依赖库
├── csv_data/                # CSV数据文件目录
├── results/                 # 结果文件目录
└── myenvs_fipynaca2.0/      # Python虚拟环境目录
```

## 安装与配置

### 创建虚拟环境

```bash
python -m venv myenvs_fipynaca2.0
myenvs_fipynaca2.0\Scripts\activate
```

### 安装依赖包

```bash
pip install -r requirements.txt
```

## 使用流程

### 阶段一：工具准备

1. **几何生成**：使用C++程序根据输入参数生成翼型坐标文件
2. **网格生成**：运行`网格生成器2_三角形.py`生成计算网格
3. **CFD求解**：运行`求解器_pvtnf.py`或`compute_Nu_f.py`进行流场和温度场计算

### 阶段二：数据生产

1. **生成采样点**：
   ```bash
   python generate_samples.py
   ```

2. **批量运行案例**：
   ```bash
   python run_all_cases.py
   ```

### 阶段三：AI训练

```bash
python train_ai_models.py
```

### 阶段四：智能寻优

```bash
python ai_optimization.py
```

### 阶段五：最终验证

使用AI找到的最优参数运行一次完整的CFD仿真进行验证。

## 脚本详细说明

### 网格生成器2_三角形.py

根据翼型坐标数据文件生成FiPy兼容的网格文件。

### 求解器系列脚本

- `求解器2.py`：基础CFD求解器，计算流场
- `求解器_pvtnf.py`：扩展求解器，同时计算流场和温度场
- `compute_Nu_f.py`：专门用于计算Nu和f参数的简化求解器

### 后处理脚本

- `post_processing.py`：基础后处理脚本，提供额外的工程信息提取
- `enhanced_post_processing.py`：增强后处理脚本，包含更详细的分析功能

后处理功能包括：
- 气动系数计算（阻力、升力系数）
- 传热分析（努塞尔数分布）
- 涡量场计算
- 边界层分析
- 压力系数分布
- 流动分离点检测
- 综合性报告生成

### 优化相关脚本

- `generate_samples.py`：使用拉丁超立方采样生成参数组合
- `run_optimization_case.py`：运行单个优化案例
- `run_all_cases.py`：批量运行所有案例
- `train_ai_models.py`：训练Nu和f预测模型
- `ai_optimization.py`：使用AI模型进行参数优化

## 参数说明

优化过程中涉及的主要参数：

- **Tt**：扰流柱顶部厚度参数
- **Ts**：扰流柱侧面厚度参数
- **Tad**：扰流柱前缘角度参数
- **Tb**：扰流柱后缘角度参数

## 输出结果

- **Nu**：努塞尔数，表征换热效率
- **f**：摩擦因子，表征流动阻力
- **Nu/(f^(1/3))**：目标参数，换热效率与流动阻力的比值

## 注意事项

1. 确保C++几何生成程序正确配置并可被调用
2. 网格生成需要Gmsh工具支持
3. 大量CFD计算可能需要较长时间
4. AI模型训练需要足够的数据点以保证准确性

## 故障排除

- 如果遇到依赖包安装问题，请检查Python版本兼容性
- 如果CFD求解发散，请检查网格质量和边界条件设置
- 如果AI优化结果不理想，请检查训练数据质量和模型参数