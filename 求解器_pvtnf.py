# %%
import matplotlib.pyplot as plt
import numpy as np
from fipy import *
from fipy.tools import numerix
from tqdm import tqdm
import os
from datetime import datetime
import pickle

# --- 检查核心库是否存在 ---
try:
    from scipy.interpolate import griddata
except ImportError:
    print("错误: 缺少 Scipy 库。")
    print("请使用 'pip install scipy' 或 'conda install scipy' 进行安装。")
    exit()

# %%
# --- 加载网格 ---
filename = "airfoil_array"  # 修改文件名以匹配
msh_filename = f"{filename}.msh2"

if not os.path.exists(msh_filename):
    print("-" * 50)
    print(f"\n错误: 未找到网格文件 '{msh_filename}'。")
    print("\n请先运行 'generate_mesh.py' 脚本来自动生成网格,")
    print("然后再重新运行此脚本。")
    print("-" * 50)
    exit()

print(f"正在从 '{msh_filename}' 加载网格...")
mesh = Gmsh2D(msh_filename)

# %%
# --- 物理参数和变量定义 (与专业范例一致) ---
# 流体属性
mu = 0.1
rho = 1.
U = 10.

# --- 新增: 热物理属性 ---
k_fluid = 2.0  # 流体导热系数
Cp_fluid = 1.0  # 流体比热容
T_inlet = 293.15  # 入口温度 (K, 例如 20°C)
T_airfoil = 353.15  # 翼型表面恒定温度 (K, 例如 80°C)

sum_res_list = []

# %%
# --- 核心变量定义 ---
Vc = mesh.cellVolumes
Vcf = CellVariable(mesh=mesh, value=Vc).faceValue

Vx = CellVariable(mesh=mesh, name="x velocity", value=U)
Vy = CellVariable(mesh=mesh, name=r"y velocity", value=0.)

Vf = FaceVariable(mesh=mesh, rank=1)
Vf.setValue((Vx.faceValue, Vy.faceValue))

p = CellVariable(mesh=mesh, name="pressure", value=0.)
pc = CellVariable(mesh=mesh, value=0.)

apx = CellVariable(mesh=mesh, value=1.)

# --- 新增: 温度场变量 ---
T = CellVariable(mesh=mesh, name="temperature", value=T_inlet)

# --- 边界条件定义 ---
inletFace = mesh.physicalFaces["inlet"]
outletFace = mesh.physicalFaces["outlet"]
airfoilsFace = mesh.physicalFaces["airfoils"]
top_bottomFace = mesh.physicalFaces["top"] | mesh.physicalFaces["bottom"]

# 流场边界条件
Vx.constrain(U, inletFace)
Vy.constrain(0., inletFace)
p.faceGrad.constrain(0., inletFace)
pc.faceGrad.constrain(0., inletFace)

Vx.faceGrad.constrain(0., outletFace)
Vy.faceGrad.constrain(0., outletFace)
p.constrain(0., outletFace)
pc.constrain(0., outletFace)

Vx.constrain(0., airfoilsFace)
Vy.constrain(0., airfoilsFace)
p.faceGrad.constrain(0., airfoilsFace)
pc.faceGrad.constrain(0., airfoilsFace)

Vx.faceGrad.constrain(0., top_bottomFace)
Vy.faceGrad.constrain(0., top_bottomFace)
p.constrain(0., top_bottomFace)
pc.constrain(0., top_bottomFace)

# --- 新增: 温度场边界条件 ---
T.constrain(T_inlet, inletFace)  # 入口温度恒定
T.constrain(T_airfoil, airfoilsFace)  # 翼型表面温度恒定
T.faceGrad.constrain(0, top_bottomFace)  # 上下壁面绝热
T.faceGrad.constrain(0, outletFace)  # 出口自由流出

# %%
# --- 方程构建 ---
# 动量方程
Vx_Eq = \
    UpwindConvectionTerm(coeff=rho * Vf, var=Vx) == \
    DiffusionTerm(coeff=mu, var=Vx) - \
    ImplicitSourceTerm(coeff=1.0, var=p.grad[0])
Vy_Eq = \
    UpwindConvectionTerm(coeff=rho * Vf, var=Vy) == \
    DiffusionTerm(coeff=mu, var=Vy) - \
    ImplicitSourceTerm(coeff=1.0, var=p.grad[1])

# 压力修正方程
# (mesh._faceAreas 带下划线是正确的)
coeff = (1. / (apx.faceValue * mesh._faceAreas * mesh._cellDistances))
pc_Eq = DiffusionTerm(coeff=coeff, var=pc) - Vf.divergence == 0

# --- 新增: 温度方程 ---
T_Eq = UpwindConvectionTerm(coeff=rho * Cp_fluid * Vf, var=T) == DiffusionTerm(coeff=k_fluid, var=T)

# %%
# --- 核心算法 ---
V_limit = 1e2
p_limit = 2e3


def OverflowPrevention():
    Vx.value[Vx.value > V_limit] = V_limit
    Vx.value[Vx.value < -V_limit] = -V_limit
    Vy.value[Vy.value > V_limit] = V_limit
    Vy.value[Vy.value < -V_limit] = -V_limit
    p.value[p.value > p_limit] = p_limit
    p.value[p.value < -p_limit] = -V_limit


def sweep(Rp, Rv):
    OverflowPrevention()
    Vx_Eq.cacheMatrix()
    xres = Vx_Eq.sweep(var=Vx, underRelaxation=Rv)
    xmat = Vx_Eq.matrix
    apx[:] = numerix.asarray(xmat.takeDiagonal())

    yres = Vy_Eq.sweep(var=Vy, underRelaxation=Rv)

    presgrad = p.grad
    facepresgrad = presgrad.faceValue
    Vf[0] = Vx.faceValue + Vcf / apx.faceValue * (presgrad[0].faceValue - facepresgrad[0])
    Vf[1] = Vy.faceValue + Vcf / apx.faceValue * (presgrad[1].faceValue - facepresgrad[1])

    pcres = pc_Eq.sweep(var=pc)

    p.setValue(p + Rp * pc)
    Vx.setValue(Vx - (Vc * pc.grad[0]) / apx)
    Vy.setValue(Vy - (Vc * pc.grad[1]) / apx)

    presgrad = p.grad
    facepresgrad = presgrad.faceValue
    Vf[0] = Vx.faceValue + Vcf / apx.faceValue * (presgrad[0].faceValue - facepresgrad[0])
    Vf[1] = Vy.faceValue + Vcf / apx.faceValue * (presgrad[1].faceValue - facepresgrad[1])
    return xres, yres, pcres


# %%
# --- 求解循环辅助函数 ---
def is_increasing(arr):
    for i in range(len(arr) - 1):
        if arr[i] >= arr[i + 1]: return False
    return True


def valueRange(val, a, b):
    return (val > a and val <= b)


# %%
# --- 求解循环 ---
# ***************************************************************
# *** 调试建议 ***
#
# 保持 MaxSweep = 5, 直到脚本从头到尾跑通，不再报任何错误。
# 跑通后再改回 300。
#
MaxSweep = 300  # <-- 调试时用 5
# MaxSweep = 300 # <-- 正式计算时用这个
# ***************************************************************

res_limit = 1e-4
sum_res = 1e10

pbar = tqdm(range(MaxSweep), desc="求解流场 (SIMPLE)")
for i in pbar:
    if sum_res > 2e2:
        Rp, Rv = 0.2, 0.5
    elif valueRange(sum_res, 28., 2e2):
        Rp, Rv = 0.3, 0.6
    elif valueRange(sum_res, 10., 28.):
        Rp, Rv = 0.5, 0.7
    elif valueRange(sum_res, 1., 10.):
        Rp, Rv = 0.7, 0.8
    else:
        Rp, Rv = 0.8, 0.9

    xres, yres, pcres = sweep(Rp, Rv)
    sum_res = sum([abs(xres), abs(yres), abs(pcres)])
    sum_res_list.append(sum_res)
    pbar.set_postfix({"sum res": f'{sum_res:.2e}', "Rp": f'{Rp:.2f}', "Rv": f'{Rv:.2f}'})

    if np.isnan(sum_res):
        print("\n错误: 计算出现无效值(NaN)，求解发散。")
        break
    if sum_res < res_limit and i > 5:  # 增加一个最小迭代次数
        print("\n残差收敛，流场求解完成。")
        break

if i == MaxSweep - 1: print("\n达到最大迭代次数，流场可能未完全收敛。")

# --- 新增: 求解温度场 ---
if 'sum_res' in locals() and not (np.isnan(sum_res) or np.isinf(sum_res)):
    print("\n开始求解温度场...")
    T_Eq.solve(var=T)
    print("温度场求解完成。")
else:
    print("\n由于流场计算失败，跳过温度场求解。")

# %%
# --- 新增: 后处理 (根据热交换器定义计算 f 和 Nu) ---
if 'sum_res' in locals() and not (np.isnan(sum_res) or np.isinf(sum_res)):
    print("\n开始进行后处理: 计算 Darcy 摩擦系数 (f) 和 Nusselt 数 (Nu)...")

    # --- 0. 共享变量和几何参数 ---

    # *********** 已修正 ***********
    # (FiPy 命名不一致: _faceAreas 带下划线, faceNormals 不带)
    faceAreas = mesh._faceAreas  # (带下划线)
    faceNormals = mesh.faceNormals  # (不带下划线)
    # ****************************

    # 获取边界掩码 (这些 .value 是正确的)
    inletFaceMask = inletFace.value
    outletFaceMask = outletFace.value
    airfoil_mask = airfoilsFace.value

    # 计算通道几何参数 (L 和 Dh)
    # L = 通道长度 (x方向)
    # H = 通道高度 (y方向)
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    L_channel = numerix.max(x_coords) - numerix.min(x_coords)
    H_channel = numerix.max(y_coords) - numerix.min(y_coords)

    # Dh = 水力直径 (对于2D通道, Dh = 2 * H)
    # 注意: 您的图片使用 D, 这里我们用 D_h
    D_h = 2 * H_channel

    print("-" * 30)
    print("几何参数 (用于 f 和 Nu):")
    print(f"  通道长度 (L_channel): {L_channel:.4f} m")
    print(f"  通道高度 (H_channel): {H_channel:.4f} m")
    print(f"  水力直径 (D_h)   : {D_h:.4f} m")
    print("-" * 30)

    # --- 1. 计算 Darcy 摩擦系数 (f) ---
    # *********** 已修正 (f 定义) ***********
    # f = (D * 2 * Delta_P) / (L * rho * V^2)

    # (a) 计算入口和出口的面积加权平均压力
    # (p.faceValue 是 FaceVariable, .value 获取其数组)
    inletPressures = p.faceValue.value[inletFaceMask]
    outletPressures = p.faceValue.value[outletFaceMask]

    # (faceAreas 是 MaskedArray, 直接用 mask 索引)
    inletAreas = faceAreas[inletFaceMask]
    outletAreas = faceAreas[outletFaceMask]

    P_inlet_avg = numerix.sum(inletPressures * inletAreas) / numerix.sum(inletAreas)
    P_outlet_avg = numerix.sum(outletPressures * outletAreas) / numerix.sum(outletAreas)



    # (b) 计算压降
    delta_P = P_inlet_avg - P_outlet_avg

    # (c) 计算 f (Darcy-Weisbach 定义)
    # 注意: 动能项使用入口速度 U
    darcy_f = (D_h * 2 * delta_P) / (L_channel * rho * U ** 2)

    print("摩擦系数 (f) 计算结果:")
    print(f"  入口平均压力 (P_inlet_avg): {P_inlet_avg:.4e} Pa")
    print(f"  出口平均压力 (P_outlet_avg): {P_outlet_avg:.4e} Pa")
    print(f"  压降 (Delta_P)        : {delta_P:.4e} Pa")
    print(f"  Darcy 摩擦系数 (f)    : {abs(darcy_f):.6f}")  # 使用绝对值
    print("-" * 30)

    # --- 2. 计算 Nusselt 数 (Nu) ---
    # Nu = (h_avg * D_h) / k_fluid
    # (注意: 您的图片用 L 作为特征长度, Nu=hL/lambda)
    # (工程上内部流动更常用 D_h。我们先用 D_h, 您可以轻易替换)

    # (a) 计算翼型表面的总传热量 Q_total
    grad_T_face = T.grad.faceValue
    # (grad_T_face.value 是数组, faceNormals 是数组)
    gradT_dot_n = numerix.sum(grad_T_face.value * faceNormals, axis=0)
    local_heat_flux = -k_fluid * gradT_dot_n
    Q_per_face = local_heat_flux * faceAreas
    Q_total = numerix.sum(Q_per_face[airfoil_mask])

    # (b) 计算翼型总湿面积 A_wetted
    A_wetted = numerix.sum(faceAreas[airfoil_mask])

    # (c) 计算出口质量加权平均温度 (Bulk Temperature)
    # T_bulk_out = (integral(T * rho * v_n * dA)) / (integral(rho * v_n * dA))
    T_outlet = T.faceValue.value[outletFaceMask]
    Vf_outlet_vector = Vf.value[..., outletFaceMask]
    normals_outlet = faceNormals[..., outletFaceMask]
    areas_outlet = faceAreas[outletFaceMask]

    v_dot_n_outlet = numerix.sum(Vf_outlet_vector * normals_outlet, axis=0)
    mass_flow_rate_per_face = rho * v_dot_n_outlet * areas_outlet

    T_weighted_numerator = numerix.sum(T_outlet * mass_flow_rate_per_face)
    total_mass_flow_rate = numerix.sum(mass_flow_rate_per_face)

    # T_outlet_avg 是质量加权的出口平均温度
    T_outlet_avg = T_weighted_numerator / total_mass_flow_rate

    # (d) 计算对数平均温差 (LMTD - Log-Mean Temperature Difference)
    # LMTD = (DeltaT_out - DeltaT_in) / ln(DeltaT_out / DeltaT_in)
    delta_T_in = T_airfoil - T_inlet
    delta_T_out = T_airfoil - T_outlet_avg

    if abs(delta_T_out - delta_T_in) < 1e-6:
        # 如果温差几乎不变 (例如纯相变), LMTD 等于任意一端温差
        delta_T_lmtd = delta_T_in
    elif delta_T_out <= 0 or (delta_T_out / delta_T_in) <= 0:
        # 增加鲁棒性: 如果出口温差为零或负(出口温度>=壁温)，LMTD无意义
        print("警告: LMTD 计算出现非正温差，可能由于出口温度等于或超过翼型温度。")
        print(f"  DeltaT_in: {delta_T_in:.2f}, DeltaT_out: {delta_T_out:.2f}")
        print("  将使用算术平均温差代替 LMTD。")
        delta_T_lmtd = (delta_T_in + delta_T_out) / 2.0
        if delta_T_lmtd <= 0:  # 极端情况
            delta_T_lmtd = delta_T_in
    else:
        # 正常 LMTD 计算
        delta_T_lmtd = (delta_T_out - delta_T_in) / numerix.log(delta_T_out / delta_T_in)

    # (e) 计算平均传热系数 h_avg
    if delta_T_lmtd == 0:
        print("错误: 平均温差为零，无法计算 h_avg。")
        h_avg = 0.0
        Nu_avg = 0.0
    else:
        # 注意：Q_total应该是绝对值，因为它是传热速率的大小
        # 同样，delta_T_lmtd也应该是绝对值，因为Nu数总是正的
        h_avg = abs(Q_total) / (A_wetted * abs(delta_T_lmtd))

        # (f) 计算平均努塞尔数 Nu_avg
        # *********** 特征长度选择 ***********
        # 您的图片使用 L: Nu = h * L / k_fluid
        # 内部流动常用 D_h: Nu = h * D_h / k_fluid
        # 我们这里使用 D_h (与 f 一致)。您可以轻易将其改为 L_channel
        L_characteristic = D_h
        # L_characteristic = L_channel # <-- 或者用这个, 匹配您图片中的 L

        Nu_avg = (h_avg * L_characteristic) / k_fluid

    print("努塞尔数 (Nu) 计算结果:")
    print(f"  翼型总散热量 (Q_total) : {Q_total:.4e} W")
    print(f"  翼型总湿面积 (A_wetted): {A_wetted:.4e} m^2")
    print(f"  入口温度 (T_inlet)     : {T_inlet:.2f} K")
    print(f"  出口平均温度 (T_outlet_avg): {T_outlet_avg:.2f} K")
    print(f"  对数平均温差 (LMTD)  : {delta_T_lmtd:.2f} K")
    print(f"  平均传热系数 (h_avg) : {h_avg:.4f} W/(m^2·K)")
    print(f"  (使用特征长度 L_c = {L_characteristic:.4f} m)")
    print(f"  平均努塞尔数 (Nu_avg)  : {Nu_avg:.4f}")
    print("-" * 30)

    # 保存求解结果供后处理使用
    def save_solution():
        """
        保存求解结果供后处理使用
        """
        print("正在保存求解结果...")
        
        # 创建保存目录
        if not os.path.exists('solver_results'):
            os.makedirs('solver_results')
            
        # 准备要保存的数据
        results = {
            'mesh': mesh,
            'Vx': Vx,
            'Vy': Vy,
            'p': p,
            'T': T,
            'Nu': Nu_avg,      # 使用正值
            'f': abs(darcy_f), # 使用正值
            'target_param': Nu_avg / (abs(darcy_f)**(1/3)) if darcy_f != 0 else 0
        }
        
        # 保存结果
        with open('solver_results/latest_solution.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("求解结果已保存到 solver_results/latest_solution.pkl")
    
    # 保存结果
    save_solution()
    
    # %%
    # --- 扩展后处理功能 ---
    print("\n开始执行扩展后处理...")
    
    # 计算速度模
    V_mag = CellVariable(mesh=mesh, name="velocity magnitude")
    V_mag.setValue(numerix.sqrt(Vx.value**2 + Vy.value**2))
    
    # 计算涡量（z方向分量）
    omega_z = CellVariable(mesh=mesh, name="vorticity (z)")
    omega_z.setValue(Vy.grad[0].value - Vx.grad[1].value)
    
    # 创建结果目录
    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S") + "_enhanced_post")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 沿中心线的速度分布（y=0）
    x_coords = mesh.cellCenters[0]
    y_coords = mesh.cellCenters[1]
    
    # 找到接近y=0的点
    center_line_indices = numerix.where(numerix.abs(y_coords.value) < 0.5)[0]
    if len(center_line_indices) > 0:
        x_center = x_coords.value[center_line_indices]
        Vx_center = Vx.value[center_line_indices]
        Vy_center = Vy.value[center_line_indices]
        Vmag_center = V_mag.value[center_line_indices]
        p_center = p.value[center_line_indices]
        T_center = T.value[center_line_indices]
        
        # 按x坐标排序
        sorted_indices = numerix.argsort(x_center)
        x_center = x_center[sorted_indices]
        Vx_center = Vx_center[sorted_indices]
        Vy_center = Vy_center[sorted_indices]
        Vmag_center = Vmag_center[sorted_indices]
        p_center = p_center[sorted_indices]
        T_center = T_center[sorted_indices]
        
        # 绘制速度分布
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(x_center, Vx_center, 'b-', label='Vx')
        plt.xlabel('x')
        plt.ylabel('Vx')
        plt.title('Streamwise Velocity Profile Along Center Line')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(x_center, Vy_center, 'r-', label='Vy')
        plt.xlabel('x')
        plt.ylabel('Vy')
        plt.title('Normal Velocity Profile Along Center Line')
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(x_center, Vmag_center, 'g-', label='|V|')
        plt.xlabel('x')
        plt.ylabel('|V|')
        plt.title('Velocity Magnitude Profile Along Center Line')
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(x_center, p_center, 'k-', label='p')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title('Pressure Profile Along Center Line')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(x_center, T_center, 'm-', label='T')
        plt.xlabel('x')
        plt.ylabel('T (K)')
        plt.title('Temperature Profile Along Center Line')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'center_line_profiles.png'))
        plt.close()
        
        # 保存数据为CSV
        data = np.column_stack((x_center, Vx_center, Vy_center, Vmag_center, p_center, T_center))
        np.savetxt(
            os.path.join(results_dir, 'center_line_profiles.csv'),
            data,
            delimiter=',',
            header='x,Vx,Vy,Vmag,pressure,temperature',
            comments=''
        )
    
    # 2. 生成带流线的云图
    try:
        # 绘制速度云图
        plt.figure(figsize=(10, 8))
        viewer = Viewer(vars=V_mag, title="Velocity Magnitude")
        viewer.plot()
        plt.savefig(os.path.join(results_dir, 'velocity_magnitude.png'))
        plt.close()
        
        # 绘制压力云图
        plt.figure(figsize=(10, 8))
        viewer = Viewer(vars=p, title="Pressure")
        viewer.plot()
        plt.savefig(os.path.join(results_dir, 'pressure.png'))
        plt.close()
        
        # 绘制温度云图
        plt.figure(figsize=(10, 8))
        viewer = Viewer(vars=T, title="Temperature")
        viewer.plot()
        plt.savefig(os.path.join(results_dir, 'temperature.png'))
        plt.close()
        
        # 绘制涡量云图
        plt.figure(figsize=(10, 8))
        viewer = Viewer(vars=omega_z, title="Vorticity")
        viewer.plot()
        plt.savefig(os.path.join(results_dir, 'vorticity.png'))
        plt.close()
        
    except Exception as e:
        print(f"绘制云图时出错: {e}")
        print("跳过云图生成...")
    
    # 3. 导出数据为CSV格式
    # 获取网格点坐标
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    
    # 创建数据矩阵
    data = np.column_stack((x_coords, y_coords, Vx.value, Vy.value, p.value, T.value))
    
    # 保存为CSV
    np.savetxt(
        os.path.join(results_dir, "field_data.csv"),
        data,
        delimiter=",",
        header="x,y,Vx,Vy,pressure,temperature",
        comments=""
    )
    
    # 4. 生成总结报告
    report_content = f"""
CFD后处理总结报告
==================

计算域信息:
-----------
- 网格单元数: {mesh.numberOfCells}
- 网格面数: {mesh.numberOfFaces}

场变量统计:
----------
- 最大速度: {numerix.max(V_mag.value):.4f}
- 最小速度: {numerix.min(V_mag.value):.4f}
- 平均速度: {numerix.average(V_mag.value):.4f}

- 最大压力: {numerix.max(p.value):.4f}
- 最小压力: {numerix.min(p.value):.4f}
- 平均压力: {numerix.average(p.value):.4f}

- 最高温度: {numerix.max(T.value):.2f} K
- 最低温度: {numerix.min(T.value):.2f} K
- 平均温度: {numerix.average(T.value):.2f} K

传热性能:
--------
- 努塞尔数 (Nu): {Nu_avg:.6f}
- 摩擦因子 (f): {abs(darcy_f):.6f}
- 目标参数 Nu/(f^(1/3)): {Nu_avg/(abs(darcy_f)**(1/3)):.6f} (如果f>0)

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 保存报告
    with open(os.path.join(results_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"扩展后处理完成，结果已保存到: {results_dir}")
    
else:
    print("\n流场计算失败，跳过力与传热计算。")

# %%
# --- 可视化 ---
if __name__ == '__main__':
    if 'sum_res' in locals() and (np.isnan(sum_res) or np.isinf(sum_res)):
        print("\n计算发散，无法生成结果图。")
        if MaxSweep < 100:
            print(" (这是调试运行，发散是正常的，请检查后处理代码是否报错)")
        input("按回车键退出。")
        exit()

    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S") + "_NS_Gmsh_Pro_Thermal")
    os.makedirs(results_dir, exist_ok=True)

    print("\n正在生成收敛历史图...")
    plt.figure()
    plt.plot(np.log10(np.array(sum_res_list)))
    plt.xlabel('sweep number');
    plt.ylabel('log10(sum of residuals)');
    plt.title('Convergence History');
    plt.grid()
    plt.savefig(os.path.join(results_dir, "convergence_history.png"));
    plt.close()

    print("\n正在生成交互式云图...")
    print("请注意: 每个Viewer窗口弹出后，程序会暂停。")
    print("请在查看完毕后手动关闭窗口，或按 'q' 键关闭，程序才会继续。")

    OverflowPrevention()

    viewer_p = Viewer(vars=p, title="Pressure Field")
    viewer_p.plot()
    viewer_p.fig.savefig(os.path.join(results_dir, "pressure_field.png"))

    viewer_vx = Viewer(vars=Vx, title="X Velocity Field")
    viewer_vx.plot()
    viewer_vx.fig.savefig(os.path.join(results_dir, "velocity_x.png"))

    viewer_vy = Viewer(vars=Vy, title="Y Velocity Field")
    viewer_vy.plot()
    viewer_vy.fig.savefig(os.path.join(results_dir, "velocity_y.png"))

    # --- 新增: 温度场可视化 ---
    viewer_T = Viewer(vars=T, title="Temperature Field (K)", datamin=T_inlet, datamax=T_airfoil)
    viewer_T.plot()
    viewer_T.fig.savefig(os.path.join(results_dir, "temperature_field.png"))

    print("\n所有图片已保存。")
    if MaxSweep < 100:
        print(f"\n*** 调试运行成功 (MaxSweep={MaxSweep})！ ***")
        print("*** 后处理代码已无语法错误。 ***")
        print("*** 现在请将 MaxSweep 改回 300 进行正式计算。 ***")

    input("所有交互式窗口已显示。按回车键退出程序...")