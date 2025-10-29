# %%
import matplotlib.pyplot as plt
import numpy as np
from fipy import *
from fipy.tools import numerix
from tqdm import tqdm
import os
from datetime import datetime

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
mu = 0.1
rho = 1.
U = 10.
sum_res_list = []

# %%
# --- 核心变量定义 (与专业范例一致) ---
Vc = mesh.cellVolumes
Vcf = CellVariable(mesh=mesh, value=Vc).faceValue

Vx = CellVariable(mesh=mesh, name="x velocity", value=U)
Vy = CellVariable(mesh=mesh, name=r"y velocity", value=0.)

Vf = FaceVariable(mesh=mesh, rank=1)
Vf.setValue((Vx.faceValue, Vy.faceValue))

p = CellVariable(mesh=mesh, name="pressure", value=0.)
pc = CellVariable(mesh=mesh, value=0.)

apx = CellVariable(mesh=mesh, value=1.)

# --- 边界条件定义 (与专业范例一致, 仅修改名称) ---
inletFace = mesh.physicalFaces["inlet"]
outletFace = mesh.physicalFaces["outlet"]
airfoilsFace = mesh.physicalFaces["airfoils"]  # <-- 此处为唯一修改
top_bottomFace = mesh.physicalFaces["top"] | mesh.physicalFaces["bottom"]

Vx.constrain(U, inletFace)
Vy.constrain(0., inletFace)
p.faceGrad.constrain(0., inletFace)
pc.faceGrad.constrain(0., inletFace)

Vx.faceGrad.constrain(0., outletFace)
Vy.faceGrad.constrain(0., outletFace)
p.constrain(0., outletFace)
pc.constrain(0., outletFace)

Vx.constrain(0., airfoilsFace)  # <-- 此处为唯一修改
Vy.constrain(0., airfoilsFace)  # <-- 此处为唯一修改
p.faceGrad.constrain(0., airfoilsFace)  # <-- 此处为唯一修改
pc.faceGrad.constrain(0., airfoilsFace)  # <-- 此处为唯一修改

Vx.faceGrad.constrain(0., top_bottomFace)
Vy.faceGrad.constrain(0., top_bottomFace)
p.constrain(0., top_bottomFace)
pc.constrain(0., top_bottomFace)

# %%
# --- 方程构建 (与专业范例完全一致) ---
Vx_Eq = \
    UpwindConvectionTerm(coeff=rho * Vf, var=Vx) == \
    DiffusionTerm(coeff=mu, var=Vx) - \
    ImplicitSourceTerm(coeff=1.0, var=p.grad[0])
Vy_Eq = \
    UpwindConvectionTerm(coeff=rho * Vf, var=Vy) == \
    DiffusionTerm(coeff=mu, var=Vy) - \
    ImplicitSourceTerm(coeff=1.0, var=p.grad[1])

coeff = (1. / (apx.faceValue * mesh._faceAreas * mesh._cellDistances))
pc_Eq = DiffusionTerm(coeff=coeff, var=pc) - Vf.divergence == 0

# %%
# --- 核心算法 (与专业范例完全一致) ---
V_limit = 1e2
p_limit = 2e3


def OverflowPrevention():
    Vx.value[Vx.value > V_limit] = V_limit
    Vx.value[Vx.value < -V_limit] = -V_limit
    Vy.value[Vy.value > V_limit] = V_limit
    Vy.value[Vy.value < -V_limit] = -V_limit
    p.value[p.value > p_limit] = p_limit
    p.value[p.value < -p_limit] = -p_limit


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
# --- 求解循环辅助函数 (与专业范例完全一致) ---
def is_increasing(arr):
    for i in range(len(arr) - 1):
        if arr[i] >= arr[i + 1]: return False
    return True


def valueRange(val, a, b):
    return (val > a and val <= b)


# %%
# --- 求解循环 (与专业范例完全一致) ---
MaxSweep = 300  # 增加迭代次数以应对更复杂的问题
res_limit = 1e-4  # 适当放宽收敛标准
sum_res = 1e10

pbar = tqdm(range(MaxSweep))
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
    # 使用更稳健的残差计算方式
    sum_res = np.sqrt(xres ** 2 + yres ** 2 + pcres ** 2)
    sum_res_list.append(sum_res)
    pbar.set_postfix({"sum res": f'{sum_res:.2e}', "Rp": f'{Rp:.2f}', "Rv": f'{Rv:.2f}'})

    if np.isnan(sum_res):
        print("\n错误: 计算出现无效值(NaN)，求解发散。")
        break
    if sum_res < res_limit:
        print("\n残差收敛，求解完成。")
        break
    if (sum_res > 1e6 and i > 50):
        print("\n错误: 残差过大且不收敛，求解发散。")
        break

if i == MaxSweep - 1: print("\n达到最大迭代次数，可能未完全收敛。")

# %%
# --- 可视化 (与专业范例完全一致) ---
if __name__ == '__main__':
    if 'sum_res' in locals() and (np.isnan(sum_res) or np.isinf(sum_res)):
        input("\n计算发散，无法生成结果图。按回车键退出。")
        exit()

    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S") + "_NS_Gmsh_Pro")
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
    print(f"收敛历史图已保存到: {os.path.join(results_dir, 'convergence_history.png')}")

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

    print("\n所有图片已保存。")
    input("所有交互式窗口已显示。按回车键退出程序...")

