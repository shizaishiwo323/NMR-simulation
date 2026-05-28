import sys
import os
import math
import json
import numpy as np

# ==========================================
# Matplotlib 显示后端补丁
# 必须放在 import matplotlib.pyplot as plt 之前。
# 如果当前环境不能弹窗，则自动退回 Agg，并保存图片。
# ==========================================
import matplotlib

_PREFERRED_BACKENDS = ["TkAgg", "QtAgg", "Qt5Agg"]
_BACKEND_SET = False
for _backend in _PREFERRED_BACKENDS:
    try:
        matplotlib.use(_backend, force=True)
        _BACKEND_SET = True
        break
    except Exception:
        pass

if not _BACKEND_SET:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from scipy.optimize import nnls
import pandas as pd

try:
    from simulation_control import SETTINGS as SIMULATION_CONTROL
except ImportError:
    SIMULATION_CONTROL = {}


def cfg_get(path, default=None):
    value = SIMULATION_CONTROL
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

# ==========================================
# 解决路径和相对导入的终极补丁
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from nmr_t2.pipelines import invert_single_signal_lcurve
except ImportError:
    invert_single_signal_lcurve = None

output_dir = cfg_get("run.output_dir")
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"Controlled output folder = {os.getcwd()}")

# ==========================================
# 0. 用户交互：选择反演模式与参数
# ==========================================
print("=" * 50)
print("欢迎使用 NMR T2 多尺度、T2-T2 与 D-T2 反演控制台")
print("=" * 50)
print("[1] 固定平滑参数模式 (适合无噪声的理想模拟数据)")
print("[2] L-Curve 自适应模式 (适合带噪声的实际/切片数据)")
if cfg_get("run.interactive", True):
    inversion_mode = input("请输入您的选择 (1 或 2): ").strip()
else:
    control_mode = str(cfg_get("inversion.mode", "fixed")).lower()
    inversion_mode = "2" if control_mode in {"2", "l_curve", "lcurve", "auto"} else "1"
    print(f"控制文件设定反演模式: {control_mode} -> {inversion_mode}")

if inversion_mode not in ['1', '2']:
    print("输入无效，默认使用 [1] 固定平滑参数模式。")
    inversion_mode = '1'

user_alpha = 1.0
if inversion_mode == '1':
    if cfg_get("run.interactive", True):
        alpha_input = input("请输入固定平滑参数 alpha 的值 (直接回车默认使用 1.0): ").strip()
        if alpha_input:
            try:
                user_alpha = float(alpha_input)
            except ValueError:
                pass
    else:
        user_alpha = float(cfg_get("inversion.alpha", 1.0))
    print(f"已设定固定平滑参数: {user_alpha}")

# 是否额外计算 D-T2 耦合前后对比
RUN_DT2_COMPARISON = bool(cfg_get("modules.dt2", True))

# 是否保留原来的 T2-T2 交换图
RUN_T2_T2 = bool(cfg_get("modules.t2_t2", True))

# ==========================================
# 1. 物理参数轴，单位：um，ms
# ==========================================
D_scaled, T2B_scaled, rho_scaled = 2.0, 3000.0, 0.005
dt = 2.0
times = np.arange(0, 1500, dt)
t2_axis = np.logspace(0, 4, 150)

# 【尺寸与体积参数】
L_large, L_small = 20.0, 8.0
depth_large, depth_small = 20.0, 8.0

area_large = (np.sqrt(3) / 4) * L_large ** 2
area_small = (np.sqrt(3) / 4) * L_small ** 2

V_large = area_large * depth_large
V_small = area_small * depth_small
total_pore_volume = V_large + V_small

H_large, H_small = L_large * np.sqrt(3) / 2, L_small * np.sqrt(3) / 2
R_in_large, R_in_small = L_large / (2 * np.sqrt(3)), L_small / (2 * np.sqrt(3))

# 连通喉道几何
L_t, W_t = 5.0, 1.0

# 数据收集容器
export_dict_t2 = {'T2_Time_ms': t2_axis}
export_dict_decay = {'Time': times}

# 图片输出目录。即使图窗不弹出，也一定会保存 PNG/PDF。
FIG_DIR = os.path.abspath("NMR_T2_Figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("Matplotlib backend =", matplotlib.get_backend())
print("Figure output folder =", FIG_DIR)
print("Simulation modules:")
print(f"  T2-T2 enabled: {RUN_T2_T2}")
print(f"  D-T2 enabled : {RUN_DT2_COMPARISON}")
print("Geometry control:")
print(f"  mode               = {cfg_get('geometry.mode', 'verified_triangle')}")
print(f"  triangle_angles_deg= {cfg_get('geometry.triangle_angles_deg', [60.0, 60.0, 60.0])}")
print(f"  contact_angle_deg  = {cfg_get('geometry.contact_angle_deg', 0.0)}")
print(f"  process            = {cfg_get('geometry.process', 'Drainage')}")


def save_figure(fig, base_name, dpi=300):
    """同时保存 PNG 和 PDF，并返回路径。"""
    png_path = os.path.join(FIG_DIR, base_name + ".png")
    pdf_path = os.path.join(FIG_DIR, base_name + ".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Figure saved: {png_path}")
    print(f"Figure saved: {pdf_path}")
    return png_path, pdf_path


def show_all_figures_or_notice():
    """交互式后端弹窗；非交互式后端只提示已保存图片。"""
    if not bool(cfg_get("run.show_figures", True)):
        print("控制文件设置 show_figures=False；不弹出图窗，只保存图片。")
        print("所有图片已经保存到：", FIG_DIR)
        return
    backend = matplotlib.get_backend().lower()
    non_interactive_keys = ["agg", "pdf", "svg", "ps", "cairo", "template"]
    if any(k in backend for k in non_interactive_keys):
        print("当前 Matplotlib 后端是非交互式后端，图窗不会弹出。")
        print("所有图片已经保存到：", FIG_DIR)
        return
    try:
        print("正在打开图窗。关闭所有图窗后程序结束。")
        plt.show(block=True)
    except TypeError:
        plt.show()
    except Exception as e:
        print("图窗显示失败，但图片已经保存。错误信息：", repr(e))
        print("图片目录：", FIG_DIR)


# ==========================================
# 2. 网格生成逻辑
# ==========================================
def create_pore_with_bubble(L, H, R_in, target_air_area, is_inverted=False, y_shift=0.0, return_geom=False):
    A_tot = (math.sqrt(3.0) / 4.0) * L ** 2
    A_w = A_tot - target_air_area
    Sw = A_w / A_tot

    if A_w < 1e-5:
        dummy = mt.createPolygon([[-0.1, y_shift], [0.1, y_shift], [0, y_shift + 0.1]])
        return dummy if return_geom else pg.Mesh(2)

    sign = -1 if is_inverted else 1
    V0, V1, V2 = [0, H * sign + y_shift], [-L / 2, y_shift], [L / 2, y_shift]
    C = [0, H / 3 * sign + y_shift]
    vertices = [V0, V1, V2]
    Sw_trans = 1.0 - math.pi / (3.0 * math.sqrt(3.0))

    if target_air_area < 1e-4:
        geom = mt.createPolygon(vertices, isClosed=True)
        for b in geom.boundaries():
            b.setMarker(1)
        return geom if return_geom else mt.createMesh(geom, area=0.15, quality=33)

    if Sw > Sw_trans + 0.001:
        geom = mt.createPolygon(vertices, isClosed=True)
        for b in geom.boundaries():
            b.setMarker(1)
        R_gas = math.sqrt(target_air_area / math.pi)
        circle_pts = [[C[0] + R_gas * math.cos(2 * math.pi * i / 60),
                       C[1] + R_gas * math.sin(2 * math.pi * i / 60)] for i in range(60)]
        circle = mt.createPolygon(circle_pts, isClosed=True)
        for b in circle.boundaries():
            b.setMarker(99)
        final_geom = geom + circle
        final_geom.addHoleMarker(C)
        return final_geom if return_geom else mt.createMesh(final_geom, area=0.15, quality=33)
    else:
        # 注意：原代码这里 area_per_corner 在使用 R_m 前尚未定义，会导致低饱和度时直接报错。
        area_per_corner = A_w / 3.0
        R_m = math.sqrt(area_per_corner / (math.sqrt(3.0) - math.pi / 3.0))
        L_c = R_m * math.sqrt(3.0)

        def norm(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def unit(v):
            n = norm(v)
            return [v[0] / n, v[1] / n]

        polys = []
        edges = [(V0, V1, V2), (V1, V0, V2), (V2, V0, V1)]
        for V_corner, V_adj1, V_adj2 in edges:
            dir1 = unit([V_adj1[0] - V_corner[0], V_adj1[1] - V_corner[1]])
            dir2 = unit([V_adj2[0] - V_corner[0], V_adj2[1] - V_corner[1]])
            P1 = [V_corner[0] + L_c * dir1[0], V_corner[1] + L_c * dir1[1]]
            P2 = [V_corner[0] + L_c * dir2[0], V_corner[1] + L_c * dir2[1]]
            dir_bisect = unit([C[0] - V_corner[0], C[1] - V_corner[1]])
            C_arc = [V_corner[0] + 2 * R_m * dir_bisect[0], V_corner[1] + 2 * R_m * dir_bisect[1]]
            angle1 = math.atan2(P1[1] - C_arc[1], P1[0] - C_arc[0])
            angle2 = math.atan2(P2[1] - C_arc[1], P2[0] - C_arc[0])
            if angle2 - angle1 > math.pi:
                angle2 -= 2 * math.pi
            elif angle1 - angle2 > math.pi:
                angle2 += 2 * math.pi
            arc_pts = [[C_arc[0] + R_m * math.cos(angle1 + (angle2 - angle1) * i / 29),
                        C_arc[1] + R_m * math.sin(angle1 + (angle2 - angle1) * i / 29)] for i in range(30)]
            poly = mt.createPolygon([V_corner] + arc_pts, isClosed=True)
            for b in poly.boundaries():
                bcen = b.center()
                dist = math.sqrt((bcen[0] - C_arc[0]) ** 2 + (bcen[1] - C_arc[1]) ** 2)
                b.setMarker(99 if abs(dist - R_m) < R_m * 0.05 else 1)
            polys.append(poly)
        final_geom = polys[0] + polys[1] + polys[2]
        return final_geom if return_geom else mt.createMesh(final_geom, area=0.15, quality=33)


# ==========================================
# 3. 求解与反演核心
# ==========================================
def _pg_solve_step(mesh, f, D, T_bulk, rho, step_dt):
    a, b = D * step_dt, -(1.0 + step_dt / T_bulk)
    bc = {'Robin': {1: rho * step_dt}}
    f = np.asarray(f, dtype=np.float64)
    try:
        return np.asarray(pg.solve(mesh, a=a, b=b, f=f, bc=bc), dtype=np.float64)
    except Exception:
        return np.asarray(pg.solver.solve(mesh, a=a, b=b, f=f, bc=bc), dtype=np.float64)


def solve_single_decay(mesh, D, T_bulk, rho, times, dt):
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    amps = []
    for _ in times:
        u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
        amps.append(np.sum(u_cell * np.asarray(mesh.cellSizes())))
        u = _pg_solve_step(mesh, u, D, T_bulk, rho, dt)
    return np.array(amps)


def build_coupled_cell_volumes(mesh):
    """
    连通系统的体积权重。
    喉道区域保留为扩散通道，但体积权重为 0，即参与扩散、不贡献信号。
    """
    cell_areas = np.asarray(mesh.cellSizes())
    centers_y = np.array([c.center().y() for c in mesh.cells()])
    cell_volumes = np.zeros_like(cell_areas)
    for i, y in enumerate(centers_y):
        if y > L_t / 2:
            cell_volumes[i] = cell_areas[i] * depth_large
        elif y < -L_t / 2:
            cell_volumes[i] = cell_areas[i] * depth_small
        else:
            cell_volumes[i] = 0.0
    return cell_volumes


def solve_coupled_decay_physics(mesh, D, T_bulk, rho, times, dt):
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    amps_tot, amps_l, amps_s = [], [], []

    centers_y = np.array([c.center().y() for c in mesh.cells()])
    idx_l = centers_y >= 0
    idx_s = centers_y < 0
    cell_volumes = build_coupled_cell_volumes(mesh)

    for _ in times:
        u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
        amps_tot.append(np.sum(u_cell * cell_volumes))
        amps_l.append(np.sum(u_cell[idx_l] * cell_volumes[idx_l]))
        amps_s.append(np.sum(u_cell[idx_s] * cell_volumes[idx_s]))
        u = _pg_solve_step(mesh, u, D, T_bulk, rho, dt)

    return np.array(amps_tot), np.array(amps_l), np.array(amps_s)


def solve_t2_t2_exchange(mesh, D, T2B, rho, times1, tm, times2):
    S = np.zeros((len(times1), len(times2)))
    for i, t1 in enumerate(times1):
        u = np.ones(mesh.nodeCount(), dtype=np.float64)
        steps_t1 = max(1, int(t1 / 2.0)) if t1 > 0 else 0
        if steps_t1 > 0:
            dt_1 = t1 / steps_t1
            for _ in range(steps_t1):
                u = _pg_solve_step(mesh, u, D, T2B, rho, dt_1)

        steps_tm = max(1, int(tm / 2.0)) if tm > 0 else 0
        if steps_tm > 0:
            dt_m = tm / steps_tm
            for _ in range(steps_tm):
                u = _pg_solve_step(mesh, u, D, T2B, rho, dt_m)

        for j, t2 in enumerate(times2):
            u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
            S[i, j] = np.sum(u_cell * np.asarray(mesh.cellSizes()))
            if j < len(times2) - 1:
                dt_2 = times2[j + 1] - times2[j]
                if dt_2 > 0:
                    u = _pg_solve_step(mesh, u, D, T2B, rho, dt_2)
    return S


def make_second_derivative_matrix(n):
    L = np.zeros((n, n))
    if n == 1:
        return L
    if n == 2:
        L[0, 0] = -1.0
        L[0, 1] = 1.0
        L[1, 0] = 1.0
        L[1, 1] = -1.0
        return L
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0
    L[0, 0] = -1.0
    L[0, 1] = 1.0
    L[-1, -2] = 1.0
    L[-1, -1] = -1.0
    return L


def invert_t2_fixed(times, signal, t2_bins, alpha):
    if np.max(signal) <= 1e-12:
        return np.zeros(len(t2_bins))
    A = np.exp(-np.outer(times, 1.0 / t2_bins))
    n = len(t2_bins)
    L = make_second_derivative_matrix(n)
    spectrum, _ = nnls(np.vstack((A, L * alpha)), np.concatenate((signal / signal[0], np.zeros(n))))
    return spectrum


def smart_invert_lcurve(times, signal, t_axis, mode, current_alpha):
    if np.max(signal) <= 1e-12:
        return np.zeros(len(t_axis))
    max_amp = np.max(signal)
    if mode == '1' or invert_single_signal_lcurve is None:
        return invert_t2_fixed(times, signal, t_axis, alpha=current_alpha) * max_amp
    elif mode == '2':
        norm_signal = signal / max_amp
        noisy_signal = norm_signal + np.random.normal(0, 0.01, len(norm_signal))
        try:
            result = invert_single_signal_lcurve(times, noisy_signal, signal_name="Sim")
            spectrum = np.asarray(result.spectrum) if hasattr(result, 'spectrum') else np.asarray(result)
            if len(spectrum) != len(t_axis):
                spectrum = np.interp(t_axis, np.logspace(0, 4, len(spectrum)), spectrum)
            return spectrum * max_amp
        except Exception:
            return invert_t2_fixed(times, signal, t_axis, alpha=current_alpha) * max_amp


def invert_t2_t2_nnls(S, times1, times2, t2_bins, alpha=1.0):
    A1 = np.exp(-np.outer(times1, 1.0 / t2_bins))
    A2 = np.exp(-np.outer(times2, 1.0 / t2_bins))
    K = np.kron(A2, A1)
    S_vec = S.flatten('F')
    n = len(t2_bins)
    L1D = make_second_derivative_matrix(n)
    I = np.eye(n)
    # 用两个方向分别平滑，避免简单相加造成抵消
    L2D = np.vstack((np.kron(I, L1D), np.kron(L1D, I)))
    A_aug = np.vstack((K, L2D * alpha))
    b_aug = np.concatenate((S_vec, np.zeros(L2D.shape[0])))
    F_vec, _ = nnls(A_aug, b_aug)
    return F_vec.reshape((n, n), order='F')


# ==========================================
# 3.1 D-T2：PFG 简化模拟 + 二维 NNLS 反演
# ==========================================
def mesh_node_coordinate(mesh, axis='x'):
    if axis.lower() == 'x':
        return np.array([n.pos().x() for n in mesh.nodes()], dtype=np.float64)
    if axis.lower() == 'y':
        return np.array([n.pos().y() for n in mesh.nodes()], dtype=np.float64)
    raise ValueError("axis must be 'x' or 'y'")


def evolve_complex_field(mesh, u_complex, duration, D, T_bulk, rho, max_dt):
    """
    复数场演化。pygimli 的 solve 通常按实数场处理，因此实部、虚部分开求。
    """
    if duration <= 1e-12:
        return u_complex
    steps = max(1, int(math.ceil(duration / max_dt)))
    h = duration / steps
    ur = np.asarray(np.real(u_complex), dtype=np.float64)
    ui = np.asarray(np.imag(u_complex), dtype=np.float64)
    for _ in range(steps):
        ur = _pg_solve_step(mesh, ur, D, T_bulk, rho, h)
        ui = _pg_solve_step(mesh, ui, D, T_bulk, rho, h)
    return ur + 1j * ui


def integrate_complex_over_cells(mesh, u_complex, cell_volumes):
    centers = mesh.cellCenters()
    ur_cell = np.asarray(pg.interpolate(mesh, np.real(u_complex), centers))
    ui_cell = np.asarray(pg.interpolate(mesh, np.imag(u_complex), centers))
    return np.sum((ur_cell + 1j * ui_cell) * cell_volumes)


def solve_dt2_signal_pfg(mesh,
                         cell_volumes,
                         D,
                         T_bulk,
                         rho,
                         te_axis,
                         b_axis,
                         dt_pde=8.0,
                         delta_diff=20.0,
                         grad_axis='x',
                         progress_name=''):
    """
    生成 D-T2 信号矩阵 S(b, TE)。

    简化物理含义：
      1) 用一对窄脉冲相位因子 exp(+i q x)、exp(-i q x) 表示 PFG 扩散编码；
      2) 两个脉冲之间用真实孔隙内扩散-表面弛豫 PDE 演化 delta_diff；
      3) 后续继续用同一 PDE 演化到不同 TE；
      4) b = q^2 * delta_diff，单位为 ms/um^2，D 单位为 um^2/ms。

    因此反演核为：
      S(b,TE) = \int\int F(D_app,T2) exp(-b D_app) exp(-TE/T2) dD_app dT2
    """
    te_axis = np.asarray(te_axis, dtype=np.float64)
    b_axis = np.asarray(b_axis, dtype=np.float64)
    if np.min(te_axis) < delta_diff:
        raise ValueError("D-T2 的最小 TE 必须 >= delta_diff；请增大 te_axis 下限或减小 delta_diff。")

    coord = mesh_node_coordinate(mesh, axis=grad_axis)
    S = np.zeros((len(b_axis), len(te_axis)), dtype=np.float64)

    for ib, bval in enumerate(b_axis):
        if progress_name:
            print(f"      {progress_name}: b [{ib + 1}/{len(b_axis)}] = {bval:.4g} ms/um^2")

        q = math.sqrt(max(float(bval), 0.0) / max(delta_diff, 1e-12))
        phase_plus = np.exp(1j * q * coord)
        phase_minus = np.conj(phase_plus)

        # 第一个梯度脉冲后，进入扩散编码期
        u = phase_plus.copy()
        u = evolve_complex_field(mesh, u, delta_diff, D, T_bulk, rho, dt_pde)

        # 第二个反向梯度脉冲
        u *= phase_minus
        current_t = delta_diff

        # 逐步演化到各个 TE，并记录信号
        for it, te in enumerate(te_axis):
            if te > current_t + 1e-12:
                u = evolve_complex_field(mesh, u, te - current_t, D, T_bulk, rho, dt_pde)
                current_t = te
            sig = integrate_complex_over_cells(mesh, u, cell_volumes)
            # 实验信号一般取回波实部；极高 b 数值下可能出现小负值，这里不强制截断，避免扭曲反演。
            S[ib, it] = np.real(sig)
    return S


def invert_d_t2_nnls(S, b_axis, te_axis, d_bins, t2_bins, alpha=0.03):
    """
    D-T2 二维非负最小二乘反演。
    S shape = (n_b, n_te)
    F shape = (n_D, n_T2)
    """
    S = np.asarray(S, dtype=np.float64)
    if np.max(np.abs(S)) <= 1e-12:
        return np.zeros((len(d_bins), len(t2_bins)))

    # 用 b=0, TE=min 的幅值归一化，反演出来的是相对谱。
    norm0 = S[0, 0] if abs(S[0, 0]) > 1e-12 else np.max(np.abs(S))
    S_norm = S / norm0
    S_norm = np.maximum(S_norm, 0.0)

    AD = np.exp(-np.outer(b_axis, d_bins))          # n_b  x n_D
    AT = np.exp(-np.outer(te_axis, 1.0 / t2_bins))  # n_te x n_T2

    # S = AD @ F @ AT.T
    # vec_F_order_F(S) = (AT kron AD) vec_F_order_F(F)
    K = np.kron(AT, AD)
    s_vec = S_norm.flatten('F')

    nD = len(d_bins)
    nT = len(t2_bins)
    LD = make_second_derivative_matrix(nD)
    LT = make_second_derivative_matrix(nT)
    L2D = np.vstack((np.kron(np.eye(nT), LD), np.kron(LT, np.eye(nD))))

    A_aug = np.vstack((K, alpha * L2D))
    b_aug = np.concatenate((s_vec, np.zeros(L2D.shape[0])))

    f_vec, _ = nnls(A_aug, b_aug)
    return f_vec.reshape((nD, nT), order='F')


def solve_uncoupled_dt2_signal(mesh_l, mesh_s, te_axis, b_axis, dt_pde, delta_diff, grad_axis):
    S_total = np.zeros((len(b_axis), len(te_axis)), dtype=np.float64)
    if mesh_l.nodeCount() > 5:
        vols_l = np.asarray(mesh_l.cellSizes()) * depth_large
        S_total += solve_dt2_signal_pfg(mesh_l, vols_l, D_scaled, T2B_scaled, rho_scaled,
                                        te_axis, b_axis, dt_pde=dt_pde, delta_diff=delta_diff,
                                        grad_axis=grad_axis, progress_name='Uncoupled Large')
    if mesh_s.nodeCount() > 5:
        vols_s = np.asarray(mesh_s.cellSizes()) * depth_small
        S_total += solve_dt2_signal_pfg(mesh_s, vols_s, D_scaled, T2B_scaled, rho_scaled,
                                        te_axis, b_axis, dt_pde=dt_pde, delta_diff=delta_diff,
                                        grad_axis=grad_axis, progress_name='Uncoupled Small')
    return S_total


def plot_dt2_map(ax, F, d_bins, t2_bins, title):
    Fp = np.asarray(F, dtype=np.float64)
    if np.max(Fp) > 1e-12:
        Fp = Fp / np.max(Fp)
    cf = ax.contourf(t2_bins, d_bins, Fp, levels=30, cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t2_bins[0], t2_bins[-1])
    ax.set_ylim(d_bins[0], d_bins[-1])
    ax.set_xlabel('T2 (ms)')
    ax.set_ylabel('D_app (um$^2$/ms)')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.25)
    return cf


# ==========================================
# 4. [第一部分] 孤立系统模拟
# ==========================================
print("\n" + "=" * 50)
print("开始执行第一部分：孤立孔隙系统 (无连通)")
print("=" * 50)

target_saturations = [float(v) for v in cfg_get("saturation.target_saturations", [0.0226, 0.12, 0.153, 0.380, 1.0])]
num_cols = len(target_saturations)

fig, axs = plt.subplots(3, num_cols, figsize=(4.5 * num_cols, 12), squeeze=False)
fig.suptitle(f"Part 1: Uncoupled T2 Dashboard (Mode {inversion_mode})", fontsize=20, fontweight='bold')

# 供后面 D-T2 耦合前后对比使用：保存 Sw=100% 的孤立网格
mesh_l_full_uncoupled = None
mesh_s_full_uncoupled = None

for i, sw in enumerate(target_saturations):
    print(f"\n[{i + 1}/{num_cols}] Processing Isolated Sw = {sw * 100:.2f}% ...")
    tw_vol = total_pore_volume * sw
    vw_s, vw_l = (tw_vol, 0.0) if tw_vol <= V_small else (V_small, tw_vol - V_small)
    al = min(area_large - (vw_l / depth_large), area_large * 0.999)
    as_ = min(area_small - (vw_s / depth_small), area_small * 0.999)

    mesh_l = create_pore_with_bubble(L_large, H_large, R_in_large, al)
    mesh_s = create_pore_with_bubble(L_small, H_small, R_in_small, as_, is_inverted=True)

    if abs(sw - 1.0) < 1e-12:
        mesh_l_full_uncoupled = mesh_l
        mesh_s_full_uncoupled = mesh_s

    for l_val, h_val, inv in [(L_large, H_large, False), (L_small, H_small, True)]:
        sign = -1 if inv else 1
        axs[0, i].plot([-l_val / 2, l_val / 2, 0, -l_val / 2], [0, 0, h_val * sign, 0], 'k-', lw=1.2)
    if mesh_l.nodeCount() > 0:
        pg.show(mesh_l, ax=axs[0, i], hold=True)
    if mesh_s.nodeCount() > 0:
        pg.show(mesh_s, ax=axs[0, i], hold=True)
    axs[0, i].set_title(f"Sw = {sw * 100:.2f}%", fontsize=14)
    axs[0, i].set_aspect('equal')
    axs[0, i].set_xlim(-12, 12)
    axs[0, i].set_ylim(-12, 20)

    amp_l_2d = solve_single_decay(mesh_l, D_scaled, T2B_scaled, rho_scaled, times, dt) if mesh_l.nodeCount() > 5 else np.zeros(len(times))
    amp_s_2d = solve_single_decay(mesh_s, D_scaled, T2B_scaled, rho_scaled, times, dt) if mesh_s.nodeCount() > 5 else np.zeros(len(times))

    amp_l_vol = amp_l_2d * depth_large
    amp_s_vol = amp_s_2d * depth_small
    at_vol = amp_l_vol + amp_s_vol

    # 1. 绘制孤立系统 1D 衰减主图
    axs[1, i].plot(times, amp_l_vol / total_pore_volume, color='#78C296', label='Large')
    axs[1, i].plot(times, amp_s_vol / total_pore_volume, color='#E59866', label='Small')
    axs[1, i].plot(times, at_vol / total_pore_volume, 'k--', label='Total')
    axs[1, i].set_ylim(0, 1.1)
    axs[1, i].grid(True, alpha=0.3)
    if i == 0:
        axs[1, i].legend()

    if i < 3:
        axins_d = axs[1, i].inset_axes([0.35, 0.40, 0.60, 0.50])
        axins_d.plot(times, amp_l_vol / total_pore_volume, color='#78C296')
        axins_d.plot(times, amp_s_vol / total_pore_volume, color='#E59866')
        axins_d.plot(times, at_vol / total_pore_volume, 'k--')
        axins_d.set_xlim(times[0], times[-1])
        max_d = np.max(at_vol / total_pore_volume)
        axins_d.set_ylim(0, max_d * 1.3 if max_d > 1e-5 else 1e-4)
        axins_d.grid(True, alpha=0.3)
        axins_d.tick_params(labelsize=8)

    spec_l = smart_invert_lcurve(times, amp_l_vol, t2_axis, inversion_mode, user_alpha) / total_pore_volume
    spec_s = smart_invert_lcurve(times, amp_s_vol, t2_axis, inversion_mode, user_alpha) / total_pore_volume
    current_total_spec = spec_l + spec_s

    # 2. 绘制孤立系统 T2 谱主图
    axs[2, i].fill_between(t2_axis, spec_l, color='#78C296', alpha=0.7)
    axs[2, i].fill_between(t2_axis, spec_s, color='#E59866', alpha=0.7)
    axs[2, i].plot(t2_axis, current_total_spec, 'k-')
    axs[2, i].set_xscale('log')
    axs[2, i].set_xlim(1, 5000)
    axs[2, i].set_ylim(0, 0.4)
    axs[2, i].grid(True, which="both", ls="--", alpha=0.3)

    if i < 3:
        axins_t = axs[2, i].inset_axes([0.35, 0.45, 0.60, 0.45])
        axins_t.fill_between(t2_axis, spec_l, color='#78C296', alpha=0.7)
        axins_t.fill_between(t2_axis, spec_s, color='#E59866', alpha=0.7)
        axins_t.plot(t2_axis, current_total_spec, 'k-')
        axins_t.set_xscale('log')
        axins_t.set_xlim(1, 5000)
        max_t = np.max(current_total_spec) if len(current_total_spec) > 0 else 0
        axins_t.set_ylim(0, max_t * 1.3 if max_t > 1e-5 else 1e-4)
        axins_t.grid(True, which="both", ls="--", alpha=0.3)
        axins_t.tick_params(labelsize=8)

    col_prefix = f'Uncoupled_Sw_{sw * 100:.2f}%'
    export_dict_decay[f'{col_prefix}_Total'] = at_vol / total_pore_volume
    export_dict_t2[f'{col_prefix}_Total'] = current_total_spec
    export_dict_t2[f'{col_prefix}_Large'] = spec_l
    export_dict_t2[f'{col_prefix}_Small'] = spec_s

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_figure(fig, "Part1_Uncoupled_T2_Dashboard")


# ==========================================
# 5. [第二部分] 连通系统与 T2-T2 谱
# ==========================================
print("\n" + "=" * 50)
print("开始执行第二部分：连通孔隙系统与 T2-T2 交换演示 (Sw=100%)")
print("=" * 50)

fig2, axs2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle("Part 2: Coupled System & T2-T2 Exchange (Sw=100%)", fontsize=20, fontweight='bold')

geom_l = create_pore_with_bubble(L_large, H_large, R_in_large, 0.0, is_inverted=False, y_shift=L_t / 2, return_geom=True)
geom_s = create_pore_with_bubble(L_small, H_small, R_in_small, 0.0, is_inverted=True, y_shift=-L_t / 2, return_geom=True)
throat = mt.createRectangle(start=[-W_t / 2, -L_t / 2], end=[W_t / 2, L_t / 2])
for b in throat.boundaries():
    b.setMarker(100)

coupled_mesh = mt.createMesh(geom_l + throat + geom_s, area=0.05, quality=34)

axs2[0].plot([-L_large / 2, L_large / 2, 0, -L_large / 2],
             [L_t / 2, L_t / 2, L_t / 2 + H_large, L_t / 2], 'k-', lw=1.2)
axs2[0].plot([-L_small / 2, L_small / 2, 0, -L_small / 2],
             [-L_t / 2, -L_t / 2, -L_t / 2 - H_small, -L_t / 2], 'k-', lw=1.2)
axs2[0].plot([-W_t / 2, W_t / 2, W_t / 2, -W_t / 2, -W_t / 2],
             [-L_t / 2, -L_t / 2, L_t / 2, L_t / 2, -L_t / 2], 'k-', lw=1.2)
pg.show(coupled_mesh, ax=axs2[0], hold=True)
axs2[0].set_title("Coupled Mesh (Throat Marker=100)")
axs2[0].set_aspect('equal')

print("   -> 正在求解连通系统 1D 衰减...")
amp_c_tot, amp_c_l, amp_c_s = solve_coupled_decay_physics(coupled_mesh, D_scaled, T2B_scaled, rho_scaled, times, dt)

norm_factor = amp_c_tot[0]

# 1. 绘制连通系统 1D 衰减主图
axs2[1].plot(times, amp_c_l / norm_factor, color='#78C296', label='Coupled Large')
axs2[1].plot(times, amp_c_s / norm_factor, color='#E59866', label='Coupled Small')
axs2[1].plot(times, amp_c_tot / norm_factor, 'k--', lw=2, label='Coupled Total')
axs2[1].set_title("1D Decay Curve")
axs2[1].legend()
axs2[1].grid(True)

print("   -> 正在反演连通系统 1D T2谱...")
unified_alpha = 0.5
spec_c_tot_raw = invert_t2_fixed(times, amp_c_tot / norm_factor, t2_axis, unified_alpha)
spec_c_tot_raw[t2_axis < 50.0] = 0.0

true_micro_ratio = amp_c_s[0] / amp_c_tot[0]
k_overlap = 5.0
log_t2 = np.log10(t2_axis)
best_valley = t2_axis[0]
min_diff = 999.0

for test_valley in t2_axis:
    weight_test = 1.0 - (1.0 / (1.0 + np.exp(-k_overlap * (log_t2 - np.log10(test_valley)))))
    area_test = np.sum(spec_c_tot_raw * weight_test) / max(np.sum(spec_c_tot_raw), 1e-12)
    if math.isnan(area_test):
        continue
    diff = abs(area_test - true_micro_ratio)
    if diff < min_diff:
        min_diff = diff
        best_valley = test_valley

weight_macro = 1.0 / (1.0 + np.exp(-k_overlap * (log_t2 - np.log10(best_valley))))
weight_micro = 1.0 - weight_macro

spec_c_l_plot = spec_c_tot_raw * weight_macro
spec_c_s_plot = spec_c_tot_raw * weight_micro

# 2. 绘制连通系统 T2 谱主图
axs2[2].fill_between(t2_axis, 0, spec_c_l_plot, color='#78C296', alpha=0.8, label='Macro-pore Mode')
axs2[2].fill_between(t2_axis, 0, spec_c_s_plot, color='#E59866', alpha=0.8, label='Micro-pore Mode')
axs2[2].plot(t2_axis, spec_c_tot_raw, 'k--', lw=2, label='Total Coupled Signal')
axs2[2].set_xscale('log')
axs2[2].set_xlim(1, 5000)
axs2[2].set_ylim(0, max(spec_c_tot_raw) * 1.2 if max(spec_c_tot_raw) > 1e-12 else 1.0)
axs2[2].set_title("1D T2 Spectrum (Coupled)")
axs2[2].legend()
axs2[2].grid(True, ls="--", alpha=0.5)

# 更新导出
export_dict_decay['Coupled_Sw_100%_Total'] = amp_c_tot / norm_factor
export_dict_t2['Coupled_Sw_100%_Total'] = spec_c_tot_raw
export_dict_t2['Coupled_Sw_100%_Large_Conceptual'] = spec_c_l_plot
export_dict_t2['Coupled_Sw_100%_Small_Conceptual'] = spec_c_s_plot

F_2d = None
t2_bins_2d = None
if RUN_T2_T2:
    print("   -> 正在执行 2D T2-T2 模拟 (运算量较大)...")
    t2t2_t_min = float(cfg_get("t2_t2.t_axis_min_ms", 1.0))
    t2t2_t_max = float(cfg_get("t2_t2.t_axis_max_ms", 10 ** 3.1))
    t2t2_bin_min = float(cfg_get("t2_t2.bin_min_ms", 10 ** 0.5))
    t2t2_bin_max = float(cfg_get("t2_t2.bin_max_ms", 10 ** 3.5))
    t1_axis_2d = np.logspace(math.log10(t2t2_t_min), math.log10(t2t2_t_max), int(cfg_get("t2_t2.t1_points", 25)))
    t2_axis_2d = np.logspace(math.log10(t2t2_t_min), math.log10(t2t2_t_max), int(cfg_get("t2_t2.t2_points", 35)))
    t2_bins_2d = np.logspace(math.log10(t2t2_bin_min), math.log10(t2t2_bin_max), int(cfg_get("t2_t2.bin_points", 35)))
    tm_mix = float(cfg_get("t2_t2.mixing_time_ms", 30.0))

    S_2d = solve_t2_t2_exchange(coupled_mesh, D_scaled, T2B_scaled, rho_scaled, t1_axis_2d, tm_mix, t2_axis_2d)
    if S_2d[0, 0] > 1e-12:
        S_2d_norm = S_2d / S_2d[0, 0]
    else:
        S_2d_norm = S_2d
    F_2d = invert_t2_t2_nnls(S_2d_norm, t1_axis_2d, t2_axis_2d, t2_bins_2d, alpha=float(cfg_get("t2_t2.alpha", 0.05)))

    cf = axs2[3].contourf(t2_bins_2d, t2_bins_2d, F_2d.T, levels=30, cmap='hot_r')
    axs2[3].plot([1, 10000], [1, 10000], 'k--', alpha=0.5)
    axs2[3].set_xscale('log')
    axs2[3].set_yscale('log')
    axs2[3].set_xlim(10, 3000)
    axs2[3].set_ylim(10, 3000)
    axs2[3].set_xlabel('T2_2 (ms)')
    axs2[3].set_ylabel('T2_1 (ms)')
    axs2[3].set_title(f"T2-T2 Map (tm={tm_mix}ms)")
else:
    axs2[3].axis('off')
    axs2[3].text(0.5, 0.5, 'T2-T2 skipped', ha='center', va='center')

plt.tight_layout()
save_figure(fig2, "Part2_Coupled_T2_T2")


# ==========================================
# 5.1 [新增] D-T2 耦合前后对比
# ==========================================
F_dt2_unc = None
F_dt2_coupled = None
d_bins_dt2 = None
t2_bins_dt2 = None
S_dt2_unc = None
S_dt2_coupled = None
b_axis_dt2 = None
te_axis_dt2 = None

if RUN_DT2_COMPARISON:
    print("\n" + "=" * 50)
    print("开始执行第三部分：D-T2 耦合前后对比 (Sw=100%)")
    print("=" * 50)

    # D-T2 信号采样轴。
    # b 单位：ms/um^2；D 单位：um^2/ms；因此 exp(-bD) 无量纲。
    # 这里故意使用较少采样点，避免计算量过大；需要更细谱图可增加数量。
    b_axis_dt2 = np.concatenate((
        np.asarray(cfg_get("dt2.b_axis", [0.0]), dtype=float),
        np.logspace(float(cfg_get("dt2.b_log_min", -3.0)),
                    float(cfg_get("dt2.b_log_max", 0.55)),
                    int(cfg_get("dt2.b_log_points", 13))),
    ))
    te_axis_dt2 = np.logspace(np.log10(float(cfg_get("dt2.te_min_ms", 25.0))),
                              np.log10(float(cfg_get("dt2.te_max_ms", 1300.0))),
                              int(cfg_get("dt2.te_points", 24)))

    # 反演谱轴
    d_bins_dt2 = np.logspace(float(cfg_get("dt2.d_log_min", -3.0)),
                             math.log10(D_scaled * float(cfg_get("dt2.d_bin_scale_to_bulk_D", 1.5))),
                             int(cfg_get("dt2.d_bins", 32)))
    t2_bins_dt2 = np.logspace(math.log10(float(cfg_get("dt2.t2_bin_min_ms", 10 ** 0.5))),
                              math.log10(float(cfg_get("dt2.t2_bin_max_ms", 10 ** 3.5))),
                              int(cfg_get("dt2.t2_bins", 40)))

    # PFG 模拟参数
    DT2_DELTA_MS = float(cfg_get("dt2.delta_ms", 20.0))       # 两个窄脉冲之间的扩散编码时间
    DT2_PDE_DT_MS = float(cfg_get("dt2.pde_dt_ms", 8.0))      # D-T2 PDE 内部步长
    DT2_GRAD_AXIS = str(cfg_get("dt2.grad_axis", "y"))        # 梯度方向
    DT2_ALPHA = float(cfg_get("dt2.alpha", 0.025))            # D-T2 二维反演平滑参数

    if mesh_l_full_uncoupled is None or mesh_s_full_uncoupled is None:
        # 兜底：如果前面没有保存到 Sw=100% 网格，就重新创建。
        mesh_l_full_uncoupled = create_pore_with_bubble(L_large, H_large, R_in_large, 0.0)
        mesh_s_full_uncoupled = create_pore_with_bubble(L_small, H_small, R_in_small, 0.0, is_inverted=True)

    print("   -> 正在生成孤立系统 D-T2 信号 S(b,TE)...")
    S_dt2_unc = solve_uncoupled_dt2_signal(mesh_l_full_uncoupled,
                                           mesh_s_full_uncoupled,
                                           te_axis_dt2,
                                           b_axis_dt2,
                                           dt_pde=DT2_PDE_DT_MS,
                                           delta_diff=DT2_DELTA_MS,
                                           grad_axis=DT2_GRAD_AXIS)

    print("   -> 正在生成连通系统 D-T2 信号 S(b,TE)...")
    vols_coupled = build_coupled_cell_volumes(coupled_mesh)
    S_dt2_coupled = solve_dt2_signal_pfg(coupled_mesh,
                                         vols_coupled,
                                         D_scaled,
                                         T2B_scaled,
                                         rho_scaled,
                                         te_axis_dt2,
                                         b_axis_dt2,
                                         dt_pde=DT2_PDE_DT_MS,
                                         delta_diff=DT2_DELTA_MS,
                                         grad_axis=DT2_GRAD_AXIS,
                                         progress_name='Coupled')

    print("   -> 正在反演孤立/连通 D-T2 二维谱...")
    F_dt2_unc = invert_d_t2_nnls(S_dt2_unc, b_axis_dt2, te_axis_dt2,
                                 d_bins_dt2, t2_bins_dt2, alpha=DT2_ALPHA)
    F_dt2_coupled = invert_d_t2_nnls(S_dt2_coupled, b_axis_dt2, te_axis_dt2,
                                     d_bins_dt2, t2_bins_dt2, alpha=DT2_ALPHA)

    # 归一化后用于投影对比
    Fu = F_dt2_unc / max(np.max(F_dt2_unc), 1e-12)
    Fc = F_dt2_coupled / max(np.max(F_dt2_coupled), 1e-12)

    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("Part 3: D-T2 Comparison Before/After Coupling (Sw=100%)", fontsize=18, fontweight='bold')

    plot_dt2_map(axs3[0], F_dt2_unc, d_bins_dt2, t2_bins_dt2, "Uncoupled D-T2")
    plot_dt2_map(axs3[1], F_dt2_coupled, d_bins_dt2, t2_bins_dt2, "Coupled D-T2")

    # 第三幅图给出 T2 边缘谱投影，便于直观看耦合导致的峰位合并/迁移。
    proj_t2_unc = np.sum(Fu, axis=0)
    proj_t2_coupled = np.sum(Fc, axis=0)
    if np.max(proj_t2_unc) > 1e-12:
        proj_t2_unc /= np.max(proj_t2_unc)
    if np.max(proj_t2_coupled) > 1e-12:
        proj_t2_coupled /= np.max(proj_t2_coupled)

    axs3[2].plot(t2_bins_dt2, proj_t2_unc, 'k-', lw=2, label='Uncoupled projection')
    axs3[2].plot(t2_bins_dt2, proj_t2_coupled, 'r--', lw=2, label='Coupled projection')
    axs3[2].set_xscale('log')
    axs3[2].set_xlabel('T2 (ms)')
    axs3[2].set_ylabel('Normalized marginal amplitude')
    axs3[2].set_title('T2 Projection from D-T2')
    axs3[2].grid(True, which='both', ls='--', alpha=0.3)
    axs3[2].legend()

    plt.tight_layout()
    save_figure(fig3, "Part3_DT2_Before_After_Coupling")


# ==========================================
# 6. 导出数据到 Excel
# ==========================================
print("\n" + "=" * 50)
print("导出数据")
print("=" * 50)

df_t2 = pd.DataFrame(export_dict_t2)
t2_file_path = os.path.abspath("T2_Inversion_Detailed_2.5D.xlsx")
df_t2.to_excel(t2_file_path, index=False)

df_decay = pd.DataFrame(export_dict_decay)
decay_file_path = os.path.abspath("Triangle_Raw_Decay.xlsx")
df_decay.to_excel(decay_file_path, index=False)

if RUN_T2_T2 and F_2d is not None and t2_bins_2d is not None:
    df_t2_t2 = pd.DataFrame(F_2d.T, index=t2_bins_2d, columns=t2_bins_2d)
    df_t2_t2.index.name = "T2_1 (ms) \\ T2_2 (ms)"
    t2_t2_file_path = os.path.abspath("T2_T2_Exchange_Map.xlsx")
    df_t2_t2.to_excel(t2_t2_file_path)
    print(f"T2-T2 map: {t2_t2_file_path}")

if RUN_DT2_COMPARISON and F_dt2_unc is not None and F_dt2_coupled is not None:
    df_dt2_unc = pd.DataFrame(F_dt2_unc, index=d_bins_dt2, columns=t2_bins_dt2)
    df_dt2_unc.index.name = "D_app (um^2/ms)"
    df_dt2_unc.columns.name = "T2 (ms)"
    dt2_unc_path = os.path.abspath("DT2_Uncoupled_Map.xlsx")
    df_dt2_unc.to_excel(dt2_unc_path)

    df_dt2_c = pd.DataFrame(F_dt2_coupled, index=d_bins_dt2, columns=t2_bins_dt2)
    df_dt2_c.index.name = "D_app (um^2/ms)"
    df_dt2_c.columns.name = "T2 (ms)"
    dt2_c_path = os.path.abspath("DT2_Coupled_Map.xlsx")
    df_dt2_c.to_excel(dt2_c_path)

    df_sig_unc = pd.DataFrame(S_dt2_unc, index=b_axis_dt2, columns=te_axis_dt2)
    df_sig_unc.index.name = "b (ms/um^2)"
    df_sig_unc.columns.name = "TE (ms)"
    sig_unc_path = os.path.abspath("DT2_Uncoupled_Signal_b_TE.xlsx")
    df_sig_unc.to_excel(sig_unc_path)

    df_sig_c = pd.DataFrame(S_dt2_coupled, index=b_axis_dt2, columns=te_axis_dt2)
    df_sig_c.index.name = "b (ms/um^2)"
    df_sig_c.columns.name = "TE (ms)"
    sig_c_path = os.path.abspath("DT2_Coupled_Signal_b_TE.xlsx")
    df_sig_c.to_excel(sig_c_path)

    print(f"D-T2 uncoupled map: {dt2_unc_path}")
    print(f"D-T2 coupled map  : {dt2_c_path}")
    print(f"D-T2 uncoupled S  : {sig_unc_path}")
    print(f"D-T2 coupled S    : {sig_c_path}")

print(f"T2 inversion: {t2_file_path}")
print(f"Raw decay   : {decay_file_path}")
control_path = os.path.abspath("Simulation_Control_Used.json")
with open(control_path, "w", encoding="utf-8") as f:
    json.dump(SIMULATION_CONTROL, f, indent=2, ensure_ascii=False)
print(f"Control file: {control_path}")
print("Figure dir  :", FIG_DIR)
print("--- Export Complete! ---")

# 最后统一显示所有图窗。
# 这样不会因为前面的 plt.show(block=False) 导致程序直接退出。
show_all_figures_or_notice()
