import numpy as np
import os
import sys

# --- 检查核心库是否存在 ---
try:
    import gmsh
except ImportError:
    print("错误: 未找到 gmsh 库。")
    print("请使用 'pip install gmsh' 进行安装。")
    exit()

# --- 全局文件名定义 ---
dat_filename = "airfoil_array_SYMMETRIC.dat"
geo_filename = "airfoil_array.geo"
msh_filename = "airfoil_array.msh2"  # 与求解器使用的 .msh2 格式保持一致


def main():
    """
    主函数，用于生成 Gmsh 网格。
    """
    if os.path.exists(msh_filename):
        print(f"网格文件 '{msh_filename}' 已存在，无需重新生成。")
        print("如果您想强制重新生成，请先手动删除此文件。")
        return

    print("开始 Gmsh 自动化流程...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # 确保输出 FiPy 兼容的 v2.2 格式

    gmsh.model.add("airfoil_array_model")
    geom = gmsh.model.geo

    # 解析翼型数据
    try:
        with open(dat_filename, 'r') as f:
            lines = f.readlines()
    except (IOError, FileNotFoundError):
        print(f"错误: 无法找到或读取 '{dat_filename}' 文件。无法生成网格。")
        gmsh.finalize()
        return

    all_airfoils = []
    current_airfoil = []
    for line in lines:
        if not line.strip():
            if current_airfoil: all_airfoils.append(np.array(current_airfoil)); current_airfoil = []
        else:
            current_airfoil.append([float(x) for x in line.split()])
    if current_airfoil: all_airfoils.append(np.array(current_airfoil))
    print(f"已从 '{dat_filename}' 文件中解析出 {len(all_airfoils)} 个翼型轮廓。")

    # 定义计算域
    all_points = np.vstack(all_airfoils)
    x_min, y_min = np.min(all_points, axis=0);
    x_max, y_max = np.max(all_points, axis=0)
    padding = 8.0
    domain_x_min, domain_y_min = x_min - padding, y_min - padding
    domain_x_max, domain_y_max = x_max + padding, y_max + padding
    lcar_airfoil = 0.05;
    lcar_domain = 2.0

    # 创建几何实体
    p1 = geom.addPoint(domain_x_min, domain_y_min, 0, lcar_domain)
    p2 = geom.addPoint(domain_x_max, domain_y_min, 0, lcar_domain)
    p3 = geom.addPoint(domain_x_max, domain_y_max, 0, lcar_domain)
    p4 = geom.addPoint(domain_x_min, domain_y_max, 0, lcar_domain)
    l_bottom = geom.addLine(p1, p2);
    l_outlet = geom.addLine(p2, p3);
    l_top = geom.addLine(p3, p4);
    l_inlet = geom.addLine(p4, p1)

    point_offset, spline_offset = 100, 100
    spline_loops = []
    for airfoil in all_airfoils:
        points = []
        _, unique_indices = np.unique(airfoil, axis=0, return_index=True)
        for x, y in airfoil[np.sort(unique_indices)]:
            points.append(geom.addPoint(x, y, 0, lcar_airfoil))
        spline_id = spline_offset + len(spline_loops)
        geom.addSpline(points + [points[0]], spline_id)
        spline_loops.append(spline_id)
        point_offset += len(points)
        spline_offset += 1

    # 创建带孔洞的平面
    outer_loop = geom.addCurveLoop([l_bottom, l_outlet, l_top, l_inlet])
    inner_loops = [geom.addCurveLoop([s]) for s in spline_loops]
    surface = geom.addPlaneSurface([outer_loop] + inner_loops)

    # 定义物理组
    geom.synchronize()
    gmsh.model.addPhysicalGroup(1, [l_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="outlet")
    gmsh.model.addPhysicalGroup(1, [l_top], name="top")
    gmsh.model.addPhysicalGroup(1, [l_bottom], name="bottom")
    gmsh.model.addPhysicalGroup(1, spline_loops, name="airfoils")
    gmsh.model.addPhysicalGroup(2, [surface], name="fluid")

    print("开始生成二维网格...")
    gmsh.model.mesh.generate(2)

    print("网格已生成。现在将打开Gmsh窗口供您预览。")
    print("请检查网格是否正确，特别是翼型周围的加密情况。")
    print("检查完毕后，请直接关闭Gmsh窗口，程序将自动保存文件。")
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.write(msh_filename)
    gmsh.finalize()
    print(f"网格文件 '{msh_filename}' 生成成功。")
    print("\n现在您可以运行 'run_solver.py' 来进行物理模拟了。")


if __name__ == '__main__':
    main()