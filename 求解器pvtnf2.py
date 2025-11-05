"""
NACA0012翼型流场和温度场耦合求解器
使用FiPy实现SIMPLE算法求解Navier-Stokes方程
"""

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

def load_mesh():
    """加载网格"""
    filename = "airfoil_array"
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
    return mesh

def setup_physical_parameters():
    """设置物理参数"""
    # 流体属性
    mu = 0.1
    rho = 1.
    U = 10.

    # 热物理属性
    k_fluid = 2.0  # 流体导热系数
    Cp_fluid = 1.0  # 流体比热容
    T_inlet = 293.15  # 入口温度 (K, 例如 20°C)
    T_airfoil = 353.15  # 翼型表面恒定温度 (K, 例如 80°C)
    
    return mu, rho, U, k_fluid, Cp_fluid, T_inlet, T_airfoil

def setup_variables_and_boundary_conditions(mesh, U, T_inlet, T_airfoil):
    """设置变量和边界条件"""
    # 核心变量定义
    Vc = mesh.cellVolumes
    Vcf = CellVariable(mesh=mesh, value=Vc).faceValue

    Vx = CellVariable(mesh=mesh, name="x velocity", value=U)
    Vy = CellVariable(mesh=mesh, name="y velocity", value=0.)

    Vf = FaceVariable(mesh=mesh, rank=1)
    Vf.setValue((Vx.faceValue, Vy.faceValue))

    p = CellVariable(mesh=mesh, name="pressure", value=0.)
    pc = CellVariable(mesh=mesh, value=0.)
    apx = CellVariable(mesh=mesh, value=1.)

    # 温度场变量
    T = CellVariable(mesh=mesh, name="temperature", value=T_inlet)

    # 边界条件定义
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

    # 温度场边界条件
    T.constrain(T_inlet, inletFace)  # 入口温度恒定
    T.constrain(T_airfoil, airfoilsFace)  # 翼型表面温度恒定
    T.faceGrad.constrain(0, top_bottomFace)  # 上下壁面绝热
    T.faceGrad.constrain(0, outletFace)  # 出口自由流出
    
    return Vc, Vcf, Vx, Vy, Vf, p, pc, apx, T, inletFace, outletFace, airfoilsFace, top_bottomFace

def build_equations(rho, mu, k_fluid, Cp_fluid, Vf, Vx, Vy, p, apx, mesh, T, pc):
    """构建控制方程"""
    # 动量方程
    Vx_Eq = UpwindConvectionTerm(coeff=rho * Vf, var=Vx) == \
            DiffusionTerm(coeff=mu, var=Vx) - \
            ImplicitSourceTerm(coeff=1.0, var=p.grad[0])
            
    Vy_Eq = UpwindConvectionTerm(coeff=rho * Vf, var=Vy) == \
            DiffusionTerm(coeff=mu, var=Vy) - \
            ImplicitSourceTerm(coeff=1.0, var=p.grad[1])

    # 压力修正方程
    coeff = (1. / (apx.faceValue * mesh._faceAreas * mesh._cellDistances))
    # 添加稳定项
    coeff *= 0.5  # 减小系数以提高稳定性
    pc_Eq = DiffusionTerm(coeff=coeff, var=pc) - Vf.divergence == 0

    # 温度方程
    T_Eq = UpwindConvectionTerm(coeff=rho * Cp_fluid * Vf, var=T) == \
           DiffusionTerm(coeff=k_fluid, var=T)
           
    return Vx_Eq, Vy_Eq, pc_Eq, T_Eq

def overflow_prevention(Vx, Vy, p, V_limit=1e2, p_limit=2e3):
    """防止变量溢出"""
    Vx.value[Vx.value > V_limit] = V_limit
    Vx.value[Vx.value < -V_limit] = -V_limit
    Vy.value[Vy.value > V_limit] = V_limit
    Vy.value[Vy.value < -V_limit] = -V_limit
    p.value[p.value > p_limit] = p_limit
    p.value[p.value < -p_limit] = -p_limit

def sweep(Vx_Eq, Vy_Eq, pc_Eq, Vx, Vy, p, pc, Vf, Vc, Vcf, apx, Rp, Rv):
    """执行一次迭代"""
    overflow_prevention(Vx, Vy, p)
    
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

def value_range(val, a, b):
    """判断值是否在范围内"""
    return (val > a and val <= b)

# --- 关键修正: 接收 L_channel, H_channel, D_h 作为参数 ---
def calculate_performance_parameters(mesh, p, T, Vf, rho, U, k_fluid, T_inlet, T_airfoil, 
                                     inletFace, outletFace, airfoilsFace,
                                     L_channel, H_channel, D_h):
    """计算性能参数（摩擦系数和努塞尔数）"""
    print("\n开始进行后处理: 计算 Darcy 摩擦系数 (f) 和 Nusselt 数 (Nu)...")
    
    # 共享变量和几何参数
    faceAreas = mesh._faceAreas  # (带下划线)
    faceNormals = mesh.faceNormals  # (不带下划线)

    # 获取边界掩码
    inletFaceMask = inletFace.value
    outletFaceMask = outletFace.value
    airfoil_mask = airfoilsFace.value

    # --- 关键修正: 几何参数已从 main 传入，无需再次计算 ---
    print("-" * 30)
    print("几何参数 (用于 f 和 Nu):")
    print(f"  通道长度 (L_channel): {L_channel:.4f} m")
    print(f"  通道高度 (H_channel): {H_channel:.4f} m")
    print(f"  水力直径 (D_h)   : {D_h:.4f} m")
    print("-" * 30)

    # 计算 Darcy 摩擦系数 (f)
    inletPressures = p.faceValue.value[inletFaceMask]
    outletPressures = p.faceValue.value[outletFaceMask]
    inletAreas = faceAreas[inletFaceMask]
    outletAreas = faceAreas[outletFaceMask]
    P_inlet_avg = numerix.sum(inletPressures * inletAreas) / numerix.sum(inletAreas)
    P_outlet_avg = numerix.sum(outletPressures * outletAreas) / numerix.sum(outletAreas)
    delta_P = P_inlet_avg - P_outlet_avg
    darcy_f = (D_h * 2 * delta_P) / (L_channel * rho * U ** 2)

    print("摩擦系数 (f) 计算结果:")
    print(f"  入口平均压力 (P_inlet_avg): {P_inlet_avg:.4e} Pa")
    print(f"  出口平均压力 (P_outlet_avg): {P_outlet_avg:.4e} Pa")
    print(f"  压降 (Delta_P)         : {delta_P:.4e} Pa")
    print(f"  Darcy 摩擦系数 (f)     : {abs(darcy_f):.6f}")
    print("-" * 30)

    # 计算 Nusselt 数 (Nu)
    grad_T_face = T.grad.faceValue
    gradT_dot_n = numerix.sum(grad_T_face.value * faceNormals, axis=0)
    local_heat_flux = -k_fluid * gradT_dot_n
    Q_per_face = local_heat_flux * faceAreas
    Q_total = numerix.sum(Q_per_face[airfoil_mask])
    A_wetted = numerix.sum(faceAreas[airfoil_mask])

    T_outlet = T.faceValue.value[outletFaceMask]
    Vf_outlet_vector = Vf.value[..., outletFaceMask]
    normals_outlet = faceNormals[..., outletFaceMask]
    areas_outlet = faceAreas[outletFaceMask]
    v_dot_n_outlet = numerix.sum(Vf_outlet_vector * normals_outlet, axis=0)
    mass_flow_rate_per_face = rho * v_dot_n_outlet * areas_outlet
    T_weighted_numerator = numerix.sum(T_outlet * mass_flow_rate_per_face)
    total_mass_flow_rate = numerix.sum(mass_flow_rate_per_face)
    T_outlet_avg = T_weighted_numerator / total_mass_flow_rate

    delta_T_in = T_airfoil - T_inlet
    delta_T_out = T_airfoil - T_outlet_avg

    if abs(delta_T_out - delta_T_in) < 1e-6:
        delta_T_lmtd = delta_T_in
    elif delta_T_out <= 0 or (delta_T_out / delta_T_in) <= 0:
        print("警告: LMTD 计算出现非正温差。")
        print(f"  DeltaT_in: {delta_T_in:.2f}, DeltaT_out: {delta_T_out:.2f}")
        print("  将使用算术平均温差代替 LMTD。")
        delta_T_lmtd = (delta_T_in + delta_T_out) / 2.0
        if delta_T_lmtd <= 0: delta_T_lmtd = delta_T_in
    else:
        delta_T_lmtd = (delta_T_out - delta_T_in) / numerix.log(delta_T_out / delta_T_in)

    if delta_T_lmtd == 0:
        print("错误: 平均温差为零，无法计算 h_avg。")
        h_avg = 0.0; Nu_avg = 0.0
    else:
        h_avg = abs(Q_total) / (A_wetted * abs(delta_T_lmtd))
        L_characteristic = D_h
        Nu_avg = (h_avg * L_characteristic) / k_fluid

    print("努塞尔数 (Nu) 计算结果:")
    print(f"  翼型总散热量 (Q_total) : {Q_total:.4e} W")
    print(f"  翼型总湿面积 (A_wetted): {A_wetted:.4e} m^2")
    print(f"  入口温度 (T_inlet)       : {T_inlet:.2f} K")
    print(f"  出口平均温度 (T_outlet_avg): {T_outlet_avg:.2f} K")
    print(f"  对数平均温差 (LMTD)   : {delta_T_lmtd:.2f} K")
    print(f"  平均传热系数 (h_avg) : {h_avg:.4f} W/(m^2·K)")
    print(f"  (使用特征长度 L_c = {L_characteristic:.4f} m)")
    print(f"  平均努塞尔数 (Nu_avg)   : {Nu_avg:.4f}")
    print("-" * 30)
    
    return abs(darcy_f), Nu_avg

def save_solution(mesh, Vx, Vy, p, T, Nu_avg, darcy_f):
    """保存求解结果"""
    print("正在保存求解结果...")
    
    if not os.path.exists('solver_results'):
        os.makedirs('solver_results')
        
    results = {
        'mesh': mesh, 'Vx': Vx, 'Vy': Vy, 'p': p, 'T': T,
        'Nu': Nu_avg, 'f': darcy_f,
        'target_param': Nu_avg / (darcy_f**(1/3)) if darcy_f > 0 else 0
    }
    
    with open('solver_results/latest_solution.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("求解结果已保存到 solver_results/latest_solution.pkl")

# --- 关键修正: 接收 L_channel, H_channel, D_h 作为参数 ---
def extended_postprocessing(mesh, Vx, Vy, p, T, Nu_avg, darcy_f, base_results_dir,
                            L_channel, H_channel, D_h):
    """扩展后处理功能"""
    print("\n开始执行扩展后处理...")
    
    V_mag = CellVariable(mesh=mesh, name="velocity magnitude")
    V_mag.setValue(numerix.sqrt(Vx.value**2 + Vy.value**2))
    
    omega_z = CellVariable(mesh=mesh, name="vorticity (z)")
    omega_z.setValue(Vy.grad[0].value - Vx.grad[1].value)
    
    results_dir = base_results_dir
    
    # 1. 沿中心线的速度分布（y=0）
    x_coords = mesh.cellCenters[0]
    y_coords = mesh.cellCenters[1]
    
    center_line_indices = numerix.where(numerix.abs(y_coords.value) < 0.5)[0]
    if len(center_line_indices) > 0:
        x_center = x_coords.value[center_line_indices]
        Vx_center = Vx.value[center_line_indices]
        Vy_center = Vy.value[center_line_indices]
        Vmag_center = V_mag.value[center_line_indices]
        p_center = p.value[center_line_indices]
        T_center = T.value[center_line_indices]
        
        sorted_indices = numerix.argsort(x_center)
        x_center = x_center[sorted_indices]
        Vx_center = Vx_center[sorted_indices]
        Vy_center = Vy_center[sorted_indices]
        Vmag_center = Vmag_center[sorted_indices]
        p_center = p_center[sorted_indices]
        T_center = T_center[sorted_indices]
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1); plt.plot(x_center, Vx_center, 'b-'); plt.xlabel('x'); plt.ylabel('Vx'); plt.title('Streamwise Velocity Profile'); plt.grid(True)
        plt.subplot(2, 3, 2); plt.plot(x_center, Vy_center, 'r-'); plt.xlabel('x'); plt.ylabel('Vy'); plt.title('Normal Velocity Profile'); plt.grid(True)
        plt.subplot(2, 3, 3); plt.plot(x_center, Vmag_center, 'g-'); plt.xlabel('x'); plt.ylabel('|V|'); plt.title('Velocity Magnitude Profile'); plt.grid(True)
        plt.subplot(2, 3, 4); plt.plot(x_center, p_center, 'k-'); plt.xlabel('x'); plt.ylabel('p'); plt.title('Pressure Profile'); plt.grid(True)
        plt.subplot(2, 3, 5); plt.plot(x_center, T_center, 'm-'); plt.xlabel('x'); plt.ylabel('T (K)'); plt.title('Temperature Profile'); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'center_line_profiles.png'))
        plt.close()
        
        data = np.column_stack((x_center, Vx_center, Vy_center, Vmag_center, p_center, T_center))
        np.savetxt(os.path.join(results_dir, 'center_line_profiles.csv'), data, delimiter=',', header='x,Vx,Vy,Vmag,pressure,temperature', comments='')
    
    # 2. 生成带流线的云图
    try:
        plt.figure(figsize=(10, 8)); viewer = Viewer(vars=V_mag, title="Velocity Magnitude"); viewer.plot(); plt.savefig(os.path.join(results_dir, 'velocity_magnitude.png')); plt.close()
        plt.figure(figsize=(10, 8)); viewer = Viewer(vars=p, title="Pressure"); viewer.plot(); plt.savefig(os.path.join(results_dir, 'pressure.png')); plt.close()
        plt.figure(figsize=(10, 8)); viewer = Viewer(vars=T, title="Temperature"); viewer.plot(); plt.savefig(os.path.join(results_dir, 'temperature.png')); plt.close()
        plt.figure(figsize=(10, 8)); viewer = Viewer(vars=omega_z, title="Vorticity"); viewer.plot(); plt.savefig(os.path.join(results_dir, 'vorticity.png')); plt.close()
    except Exception as e:
        print(f"绘制云图时出错: {e}")
        print("跳过云图生成...")
    
    # 3. 导出数据为CSV格式
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    data = np.column_stack((x_coords, y_coords, Vx.value, Vy.value, p.value, T.value))
    np.savetxt(os.path.join(results_dir, "field_data.csv"), data, delimiter=",", header="x,y,Vx,Vy,pressure,temperature", comments="")
    
    # 4. 生成总结报告
    # --- 关键修正: 这里的变量 L_channel, H_channel, D_h 现在可以被正确访问 ---
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
- 摩擦因子 (f): {darcy_f:.6f}
- 目标参数 Nu/(f^(1/3)): {Nu_avg/(darcy_f**(1/3)):.6f} (如果f>0)

几何参数:
--------
- 通道长度 (L_channel): {L_channel:.4f} m
- 通道高度 (H_channel): {H_channel:.4f} m
- 水力直径 (D_h): {D_h:.4f} m

(报告中使用的Nu和f的详细计算过程见 calculate_performance_parameters 函数的打印输出)

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(os.path.join(results_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"扩展后处理完成，结果已保存到: {results_dir}")

def visualize_results(mesh, p, Vx, Vy, T, sum_res_list, MaxSweep, sum_res, T_inlet, T_airfoil, results_dir):
    """可视化结果"""
    if 'sum_res' in locals() and (np.isnan(sum_res) or np.isinf(sum_res)):
        print("\n计算发散，无法生成结果图。")
        if MaxSweep < 100:
            print(" (这是调试运行，发散是正常的，请检查后处理代码是否报错)")
        input("按回车键退出。")
        exit()

    # 收敛历史图
    print("\n正在生成收敛历史图...")
    plt.figure()
    plt.plot(np.log10(np.array(sum_res_list)))
    plt.xlabel('sweep number')
    plt.ylabel('log10(sum of residuals)')
    plt.title('Convergence History')
    plt.grid()
    plt.savefig(os.path.join(results_dir, "convergence_history.png"))
    plt.close()

    print("\n正在生成交互式云图...")
    print("请注意: 每个Viewer窗口弹出后，程序会暂停。")
    print("请在查看完毕后手动关闭窗口，或按 'q' 键关闭，程序才会继续。")

    overflow_prevention(Vx, Vy, p)

    viewer_p = Viewer(vars=p, title="Pressure Field")
    viewer_p.plot()
    viewer_p.fig.savefig(os.path.join(results_dir, "pressure_field.png"))

    viewer_vx = Viewer(vars=Vx, title="X Velocity Field")
    viewer_vx.plot()
    viewer_vx.fig.savefig(os.path.join(results_dir, "velocity_x.png"))

    viewer_vy = Viewer(vars=Vy, title="Y Velocity Field")
    viewer_vy.plot()
    viewer_vy.fig.savefig(os.path.join(results_dir, "velocity_y.png"))

    # 温度场可视化
    viewer_T = Viewer(vars=T, title="Temperature Field (K)", datamin=T_inlet, datamax=T_airfoil)
    viewer_T.plot()
    viewer_T.fig.savefig(os.path.join(results_dir, "temperature_field.png"))

    print("\n所有图片已保存。")
    if MaxSweep < 100:
        print(f"\n*** 调试运行成功 (MaxSweep={MaxSweep})！ ***")
        print("*** 后处理代码已无语法错误。 ***")
        print("*** 现在请将 MaxSweep 改回 300 进行正式计算。 ***")

    input("所有交互式窗口已显示。按回车键退出程序...")

def main():
    """主函数"""
    mesh = load_mesh()
    
    mu, rho, U, k_fluid, Cp_fluid, T_inlet, T_airfoil = setup_physical_parameters()
    
    Vc, Vcf, Vx, Vy, Vf, p, pc, apx, T, inletFace, outletFace, airfoilsFace, top_bottomFace = \
        setup_variables_and_boundary_conditions(mesh, U, T_inlet, T_airfoil)
    
    Vx_Eq, Vy_Eq, pc_Eq, T_Eq = build_equations(rho, mu, k_fluid, Cp_fluid, Vf, Vx, Vy, p, apx, mesh, T, pc)
    
    MaxSweep = 100  # 正式计算时使用300
    res_limit = 1e-4
    sum_res_list = []
    sum_res = 1e10

    pbar = tqdm(range(MaxSweep), desc="求解流场 (SIMPLE)")
    for i in pbar:
        if sum_res > 1e4: Rp, Rv = 0.3, 0.6
        elif sum_res > 1e3: Rp, Rv = 0.4, 0.7
        elif sum_res > 1e2: Rp, Rv = 0.5, 0.8
        elif sum_res > 1e1: Rp, Rv = 0.6, 0.9
        else: Rp, Rv = 0.7, 0.95
        
        xres, yres, pcres = sweep(Vx_Eq, Vy_Eq, pc_Eq, Vx, Vy, p, pc, Vf, Vc, Vcf, apx, Rp, Rv)
        sum_res = sum([abs(xres), abs(yres), abs(pcres)])
        sum_res_list.append(sum_res)
        pbar.set_postfix({"sum res": f'{sum_res:.2e}', "Rp": f'{Rp:.2f}', "Rv": f'{Rv:.2f}'})

        if np.isnan(sum_res):
            print("\n错误: 计算出现无效值(NaN)，求解发散。")
            break
        if sum_res < res_limit and i > 5:
            print("\n残差收敛，流场求解完成。")
            break

    if i == MaxSweep - 1: 
        print("\n达到最大迭代次数，流场可能未完全收敛。")

    if 'sum_res' in locals() and not (np.isnan(sum_res) or np.isinf(sum_res)):
        print("\n开始求解温度场...")
        T_Eq.solve(var=T)
        print("温度场求解完成。")
    else:
        print("\n由于流场计算失败，跳过温度场求解。")

    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S") + "_cfd_solution")
    os.makedirs(results_dir, exist_ok=True)

    # --- 关键修正: 在 main 函数中计算一次几何参数 ---
    print("\n计算几何参数...")
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    L_channel = numerix.max(x_coords) - numerix.min(x_coords)
    H_channel = numerix.max(y_coords) - numerix.min(y_coords)
    D_h = 2 * H_channel
    print(f"  - 通道长度 (L_channel): {L_channel:.4f} m")
    print(f"  - 通道高度 (H_channel): {H_channel:.4f} m")

    if 'sum_res' in locals() and not (np.isnan(sum_res) or np.isinf(sum_res)):
        # --- 关键修正: 将几何参数传递下去 ---
        darcy_f, Nu_avg = calculate_performance_parameters(
            mesh, p, T, Vf, rho, U, k_fluid, T_inlet, T_airfoil, 
            inletFace, outletFace, airfoilsFace,
            L_channel, H_channel, D_h # <-- 新增
        )
        
        save_solution(mesh, Vx, Vy, p, T, Nu_avg, darcy_f)
        
        # --- 关键修正: 将几何参数传递下去 ---
        extended_postprocessing(mesh, Vx, Vy, p, T, Nu_avg, darcy_f, results_dir,
                                L_channel, H_channel, D_h # <-- 新增
        )
    else:
        print("\n流场计算失败，跳过力与传热计算。")

    visualize_results(mesh, p, Vx, Vy, T, sum_res_list, MaxSweep, sum_res, T_inlet, T_airfoil, results_dir)

if __name__ == '__main__':
    main()