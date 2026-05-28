import sys
import os

# ==========================================
# 解决路径和相对导入的终极补丁
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import math
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from scipy.optimize import nnls
import pandas as pd

from nmr_t2.pipelines import invert_single_signal_lcurve

# ==========================================
# 0. 用户交互：选择反演模式与参数
# ==========================================
print("=" * 50)
print("欢迎使用 NMR T1/T2 多尺度及二维交换反演控制台")
print("=" * 50)
print("[1] 固定平滑参数模式 (适合无噪声的理想模拟数据)")
print("[2] L-Curve 自适应模式 (适合带噪声的实际/切片数据)")
inversion_mode = input("请输入您的选择 (1 或 2): ").strip()

if inversion_mode not in ['1', '2']:
    print("输入无效，默认使用 [1] 固定平滑参数模式。")
    inversion_mode = '1'

user_alpha = 0.2
if inversion_mode == '1':
    alpha_input = input("请输入固定平滑参数 alpha 的值 (直接回车默认使用 0.2): ").strip()
    if alpha_input:
        try:
            user_alpha = float(alpha_input)
        except ValueError:
            pass
    print(f"已设定固定平滑参数: {user_alpha}")

# ==========================================
# 1. 物理参数轴，单位：um，ms
# ==========================================
D_scaled, dt = 2.0, 2.0
times = np.arange(0, 1500, dt)

# T2 物理参数
T2B_scaled, rho2_scaled = 3000.0, 0.005
t2_axis = np.logspace(0, 4, 150)

# [新增] T1 物理参数 (通常 T1 弛豫慢于 T2，设定比值为 1.5)
T1_T2_RATIO = 1.5
T1B_scaled = T2B_scaled * T1_T2_RATIO
rho1_scaled = rho2_scaled / T1_T2_RATIO
t1_axis = np.logspace(0, 4, 150)

# 几何参数
L_large, L_small = 20.0, 8.0
depth_large, depth_small = 20.0, 8.0

area_large = (np.sqrt(3) / 4) * L_large ** 2
area_small = (np.sqrt(3) / 4) * L_small ** 2

V_large = area_large * depth_large
V_small = area_small * depth_small
total_pore_volume = V_large + V_small

H_large, H_small = L_large * np.sqrt(3) / 2, L_small * np.sqrt(3) / 2
R_in_large, R_in_small = L_large / (2 * np.sqrt(3)), L_small / (2 * np.sqrt(3))

# 喉道参数 (进入中速交换区)
L_t, W_t = 10.0, 0.2

# ==========================================
# 数据收集容器
# ==========================================
export_dict_t2 = {'T2_Time_ms': t2_axis}
export_dict_t1 = {'T1_Time_ms': t1_axis}  # [新增] T1 导出字典
export_dict_decay = {'Time': times}


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

    if not is_inverted:
        V0, V1, V2 = [0, H + y_shift], [-L / 2, y_shift], [L / 2, y_shift]
        C = [0, H / 3 + y_shift]
    else:
        V0, V1, V2 = [0, -H + y_shift], [-L / 2, y_shift], [L / 2, y_shift]
        C = [0, -H / 3 + y_shift]

    vertices = [V0, V1, V2]
    Sw_trans = 1.0 - math.pi / (3.0 * math.sqrt(3.0))

    if target_air_area < 1e-4:
        geom = mt.createPolygon(vertices, isClosed=True)
        for b in geom.boundaries(): b.setMarker(1)
        return geom if return_geom else mt.createMesh(geom, area=0.15, quality=33)

    if Sw > Sw_trans + 0.001:
        geom = mt.createPolygon(vertices, isClosed=True)
        for b in geom.boundaries(): b.setMarker(1)
        R_gas = math.sqrt(target_air_area / math.pi)
        circle_pts = [[C[0] + R_gas * math.cos(2 * math.pi * i / 60), C[1] + R_gas * math.sin(2 * math.pi * i / 60)] for
                      i in range(60)]
        circle = mt.createPolygon(circle_pts, isClosed=True)
        for b in circle.boundaries(): b.setMarker(99)
        final_geom = geom + circle
        final_geom.addHoleMarker(C)
        return final_geom if return_geom else mt.createMesh(final_geom, area=0.15, quality=33)

    else:
        area_per_corner = A_w / 3.0
        R_m = math.sqrt(area_per_corner / (math.sqrt(3.0) - math.pi / 3.0))
        L_c = R_m * math.sqrt(3.0)

        def norm(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def unit(v):
            n = norm(v); return [v[0] / n, v[1] / n]

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
def solve_single_decay(mesh, D, T_bulk, rho, times, dt):
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    amps = []
    a, b = D * dt, -(1.0 + dt / T_bulk)
    bc = {'Robin': {1: rho * dt}}
    for t in times:
        u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
        amps.append(np.sum(u_cell * np.asarray(mesh.cellSizes())))
        u_arr = np.asarray(u, dtype=np.float64)
        try:
            u = pg.solve(mesh, a=a, b=b, f=u_arr, bc=bc)
        except:
            u = pg.solver.solve(mesh, a=a, b=b, f=u_arr, bc=bc)
    return np.array(amps)


def solve_coupled_decay_components(mesh, D, T_bulk, rho, times, dt):
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    amps_tot, amps_l, amps_s = [], [], []
    a, b = D * dt, -(1.0 + dt / T_bulk)
    bc = {'Robin': {1: rho * dt}}

    cell_areas = np.asarray(mesh.cellSizes())
    centers_y = np.array([c.center().y() for c in mesh.cells()])
    idx_l = centers_y >= 0
    idx_s = centers_y < 0

    for t in times:
        u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
        amps_tot.append(np.sum(u_cell * cell_areas))
        amps_l.append(np.sum(u_cell[idx_l] * cell_areas[idx_l]))
        amps_s.append(np.sum(u_cell[idx_s] * cell_areas[idx_s]))
        u_arr = np.asarray(u, dtype=np.float64)
        try:
            u = pg.solve(mesh, a=a, b=b, f=u_arr, bc=bc)
        except:
            u = pg.solver.solve(mesh, a=a, b=b, f=u_arr, bc=bc)
    return np.array(amps_tot), np.array(amps_l), np.array(amps_s)


def solve_t2_t2_exchange(mesh, D, T2B, rho, times1, tm, times2):
    S = np.zeros((len(times1), len(times2)))
    a_mix, b_mix = D * 1.0, -1.0
    bc_mix = {'Neumann': {1: 0.0}}

    for i, t1 in enumerate(times1):
        u = np.ones(mesh.nodeCount(), dtype=np.float64)
        steps_t1 = max(1, int(t1)) if t1 > 0 else 0
        if steps_t1 > 0:
            a_t1, b_t1 = D * (t1 / steps_t1), -(1.0 + (t1 / steps_t1) / T2B)
            bc_t1 = {'Robin': {1: rho * (t1 / steps_t1)}}
            for _ in range(steps_t1): u = pg.solve(mesh, a=a_t1, b=b_t1, f=u, bc=bc_t1)

        steps_tm = max(1, int(tm)) if tm > 0 else 0
        if steps_tm > 0:
            a_tm = D * (tm / steps_tm)
            for _ in range(steps_tm): u = pg.solve(mesh, a=a_tm, b=b_mix, f=u, bc=bc_mix)

        for j, t2 in enumerate(times2):
            u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
            S[i, j] = np.sum(u_cell * np.asarray(mesh.cellSizes()))
            if j < len(times2) - 1:
                dt_2 = times2[j + 1] - times2[j]
                if dt_2 > 0:
                    a_t2, b_t2 = D * dt_2, -(1.0 + dt_2 / T2B)
                    bc_t2 = {'Robin': {1: rho * dt_2}}
                    u = pg.solve(mesh, a=a_t2, b=b_t2, f=u, bc=bc_t2)
    return S


def invert_t_fixed(times, signal, t_bins, alpha):
    if np.max(signal) <= 1e-5: return np.zeros(len(t_bins))
    A = np.exp(-np.outer(times, 1.0 / t_bins))
    n = len(t_bins)
    L = np.zeros((n, n))
    [L.__setitem__((i, i - 1), 1) or L.__setitem__((i, i), -2) or L.__setitem__((i, i + 1), 1) for i in range(1, n - 1)]
    L[0, 0] = -1;
    L[0, 1] = 1;
    L[-1, -2] = 1;
    L[-1, -1] = -1
    spectrum, _ = nnls(np.vstack((A, L * alpha)), np.concatenate((signal / signal[0], np.zeros(n))))
    return spectrum


def smart_invert_lcurve(times, signal, t_axis, mode, current_alpha):
    if np.max(signal) <= 1e-5: return np.zeros(len(t_axis))
    max_amp = np.max(signal)
    if mode == '1':
        return invert_t_fixed(times, signal, t_axis, alpha=current_alpha) * max_amp
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
            return invert_t_fixed(times, signal, t_axis, alpha=current_alpha) * max_amp


def invert_t2_t2_nnls(S, times1, times2, t2_bins, alpha=1.0):
    A1 = np.exp(-np.outer(times1, 1.0 / t2_bins))
    A2 = np.exp(-np.outer(times2, 1.0 / t2_bins))
    K = np.kron(A2, A1)
    S_vec = S.flatten('F')
    n = len(t2_bins)
    L1D = np.zeros((n, n))
    for i in range(1, n - 1): L1D[i, i - 1] = 1; L1D[i, i] = -2; L1D[i, i + 1] = 1
    L1D[0, 0] = -1;
    L1D[0, 1] = 1;
    L1D[-1, -2] = 1;
    L1D[-1, -1] = -1
    I = np.eye(n)
    L2D = np.kron(I, L1D) + np.kron(L1D, I)
    A_aug = np.vstack((K, L2D * alpha))
    b_aug = np.concatenate((S_vec, np.zeros(L2D.shape[0])))
    F_vec, _ = nnls(A_aug, b_aug)
    return F_vec.reshape((n, n), order='F')


# ==========================================
# 4. [第一部分] 孤立系统模拟 (同时计算 T2 和 T1)
# ==========================================
print("\n" + "=" * 50)
print("开始执行第一部分：孤立孔隙系统的 T2 和 T1 模拟")
print("=" * 50)

target_saturations = [0.0226, 0.12, 0.153, 0.380, 1]
num_cols = len(target_saturations)

# [画板初始化] 为 T2 和 T1 分别准备画板
fig_t2, axs_t2 = plt.subplots(3, num_cols, figsize=(4.5 * num_cols, 12), squeeze=False)
fig_t2.suptitle(f"Part 1.A: Uncoupled T2 Dashboard (Mode {inversion_mode})", fontsize=20, fontweight='bold')

fig_t1, axs_t1 = plt.subplots(3, num_cols, figsize=(4.5 * num_cols, 12), squeeze=False)
fig_t1.suptitle(f"Part 1.B: Uncoupled T1 Dashboard (Mode {inversion_mode})", fontsize=20, fontweight='bold')

for i, sw in enumerate(target_saturations):
    print(f"\n[{i + 1}/{num_cols}] Processing Isolated Sw = {sw * 100:.2f}% ...")
    tw_vol = total_pore_volume * sw
    vw_s, vw_l = (tw_vol, 0.0) if tw_vol <= V_small else (V_small, tw_vol - V_small)
    al = min(area_large - (vw_l / depth_large), area_large * 0.999)
    as_ = min(area_small - (vw_s / depth_small), area_small * 0.999)

    # 共用几何网格
    mesh_l = create_pore_with_bubble(L_large, H_large, R_in_large, al)
    mesh_s = create_pore_with_bubble(L_small, H_small, R_in_small, as_, is_inverted=True)

    # ---------------- 绘制 T2 与 T1 的几何分布 ----------------
    for fig_axs in [axs_t2, axs_t1]:
        for l_val, h_val, inv in [(L_large, H_large, False), (L_small, H_small, True)]:
            sign = -1 if inv else 1
            fig_axs[0, i].plot([-l_val / 2, l_val / 2, 0, -l_val / 2], [0, 0, h_val * sign, 0], 'k-', lw=1.2)
        if mesh_l.nodeCount() > 0: pg.show(mesh_l, ax=fig_axs[0, i], hold=True)
        if mesh_s.nodeCount() > 0: pg.show(mesh_s, ax=fig_axs[0, i], hold=True)
        fig_axs[0, i].set_title(f"Sw = {sw * 100:.2f}%", fontsize=14)
        fig_axs[0, i].set_aspect('equal');
        fig_axs[0, i].set_xlim(-12, 12);
        fig_axs[0, i].set_ylim(-12, 20)

    # ---------------- 计算与反演 T2 ----------------
    amp_l_2d_t2 = solve_single_decay(mesh_l, D_scaled, T2B_scaled, rho2_scaled, times,
                                     dt) if mesh_l.nodeCount() > 5 else np.zeros(len(times))
    amp_s_2d_t2 = solve_single_decay(mesh_s, D_scaled, T2B_scaled, rho2_scaled, times,
                                     dt) if mesh_s.nodeCount() > 5 else np.zeros(len(times))

    amp_l_vol_t2 = amp_l_2d_t2 * depth_large
    amp_s_vol_t2 = amp_s_2d_t2 * depth_small
    at_vol_t2 = amp_l_vol_t2 + amp_s_vol_t2

    axs_t2[1, i].plot(times, amp_l_vol_t2 / total_pore_volume, color='#78C296', label='Large')
    axs_t2[1, i].plot(times, amp_s_vol_t2 / total_pore_volume, color='#E59866', label='Small')
    axs_t2[1, i].plot(times, at_vol_t2 / total_pore_volume, 'k--', label='Total')
    axs_t2[1, i].set_ylim(0, 1.1);
    axs_t2[1, i].grid(True, alpha=0.3)
    if i == 0: axs_t2[1, i].legend()

    spec_l_t2 = smart_invert_lcurve(times, amp_l_vol_t2, t2_axis, inversion_mode, user_alpha) / total_pore_volume
    spec_s_t2 = smart_invert_lcurve(times, amp_s_vol_t2, t2_axis, inversion_mode, user_alpha) / total_pore_volume
    current_total_spec_t2 = spec_l_t2 + spec_s_t2

    axs_t2[2, i].fill_between(t2_axis, spec_l_t2, color='#78C296', alpha=0.7)
    axs_t2[2, i].fill_between(t2_axis, spec_s_t2, color='#E59866', alpha=0.7)
    axs_t2[2, i].plot(t2_axis, current_total_spec_t2, 'k-')
    axs_t2[2, i].set_xscale('log');
    axs_t2[2, i].set_xlim(1, 5000);
    axs_t2[2, i].set_ylim(0, 0.4)
    axs_t2[2, i].grid(True, which="both", ls="--", alpha=0.3)

    # ---------------- 计算与反演 T1 ----------------
    amp_l_2d_t1 = solve_single_decay(mesh_l, D_scaled, T1B_scaled, rho1_scaled, times,
                                     dt) if mesh_l.nodeCount() > 5 else np.zeros(len(times))
    amp_s_2d_t1 = solve_single_decay(mesh_s, D_scaled, T1B_scaled, rho1_scaled, times,
                                     dt) if mesh_s.nodeCount() > 5 else np.zeros(len(times))

    amp_l_vol_t1 = amp_l_2d_t1 * depth_large
    amp_s_vol_t1 = amp_s_2d_t1 * depth_small
    at_vol_t1 = amp_l_vol_t1 + amp_s_vol_t1

    axs_t1[1, i].plot(times, amp_l_vol_t1 / total_pore_volume, color='#78C296', label='Large')
    axs_t1[1, i].plot(times, amp_s_vol_t1 / total_pore_volume, color='#E59866', label='Small')
    axs_t1[1, i].plot(times, at_vol_t1 / total_pore_volume, 'k--', label='Total')
    axs_t1[1, i].set_ylim(0, 1.1);
    axs_t1[1, i].grid(True, alpha=0.3)
    if i == 0: axs_t1[1, i].legend()

    spec_l_t1 = smart_invert_lcurve(times, amp_l_vol_t1, t1_axis, inversion_mode, user_alpha) / total_pore_volume
    spec_s_t1 = smart_invert_lcurve(times, amp_s_vol_t1, t1_axis, inversion_mode, user_alpha) / total_pore_volume
    current_total_spec_t1 = spec_l_t1 + spec_s_t1

    axs_t1[2, i].fill_between(t1_axis, spec_l_t1, color='#78C296', alpha=0.7)
    axs_t1[2, i].fill_between(t1_axis, spec_s_t1, color='#E59866', alpha=0.7)
    axs_t1[2, i].plot(t1_axis, current_total_spec_t1, 'k-')
    axs_t1[2, i].set_xscale('log');
    axs_t1[2, i].set_xlim(1, 5000);
    axs_t1[2, i].set_ylim(0, 0.4)
    axs_t1[2, i].grid(True, which="both", ls="--", alpha=0.3)

    # ---------------- 记录数据 ----------------
    col_prefix = f'Uncoupled_Sw_{sw * 100:.2f}%'
    export_dict_decay[f'{col_prefix}_T2_Total'] = at_vol_t2 / total_pore_volume
    export_dict_decay[f'{col_prefix}_T1_Total'] = at_vol_t1 / total_pore_volume

    export_dict_t2[f'{col_prefix}_Total'] = current_total_spec_t2
    export_dict_t2[f'{col_prefix}_Large'] = spec_l_t2
    export_dict_t2[f'{col_prefix}_Small'] = spec_s_t2

    export_dict_t1[f'{col_prefix}_Total'] = current_total_spec_t1
    export_dict_t1[f'{col_prefix}_Large'] = spec_l_t1
    export_dict_t1[f'{col_prefix}_Small'] = spec_s_t1

fig_t2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_t1.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_t2.show()
fig_t1.show()

# ==========================================
# 5. [第二部分] 连通系统与 T2-T2 谱 (100% 饱和)
# ==========================================
print("\n" + "=" * 50)
print("开始执行第二部分：连通孔隙系统与 T2-T2 交换演示 (Sw=100%)")
print("=" * 50)

fig2, axs2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle("Part 2: Coupled System & T2-T2 Exchange (Sw=100%)", fontsize=20, fontweight='bold')

geom_l = create_pore_with_bubble(L_large, H_large, R_in_large, 0.0, is_inverted=False, y_shift=L_t / 2,
                                 return_geom=True)
geom_s = create_pore_with_bubble(L_small, H_small, R_in_small, 0.0, is_inverted=True, y_shift=-L_t / 2,
                                 return_geom=True)
throat = mt.createRectangle(start=[-W_t / 2, -L_t / 2], end=[W_t / 2, L_t / 2])
for b in throat.boundaries(): b.setMarker(1)

coupled_geom = geom_l + throat + geom_s
coupled_mesh = mt.createMesh(coupled_geom, area=0.05, quality=34)

axs2[0].plot([-L_large / 2, L_large / 2, 0, -L_large / 2], [L_t / 2, L_t / 2, L_t / 2 + H_large, L_t / 2], 'k-', lw=1.2)
axs2[0].plot([-L_small / 2, L_small / 2, 0, -L_small / 2], [-L_t / 2, -L_t / 2, -L_t / 2 - H_small, -L_t / 2], 'k-',
             lw=1.2)
axs2[0].plot([-W_t / 2, W_t / 2, W_t / 2, -W_t / 2, -W_t / 2], [-L_t / 2, -L_t / 2, L_t / 2, L_t / 2, -L_t / 2], 'k-',
             lw=1.2)
pg.show(coupled_mesh, ax=axs2[0], hold=True)
axs2[0].set_title("Coupled Mesh with Throat");
axs2[0].set_aspect('equal')

print("   -> 正在求解连通系统 1D 衰减 (提取大小孔分量)...")
amp_c_tot, amp_c_l, amp_c_s = solve_coupled_decay_components(coupled_mesh, D_scaled, T2B_scaled, rho2_scaled, times, dt)

norm_factor = amp_c_tot[0]
axs2[1].plot(times, amp_c_l / norm_factor, color='#78C296', label='Coupled Large')
axs2[1].plot(times, amp_c_s / norm_factor, color='#E59866', label='Coupled Small')
axs2[1].plot(times, amp_c_tot / norm_factor, 'k--', lw=2, label='Coupled Total')
axs2[1].set_title("1D Decay Curve")
axs2[1].legend();
axs2[1].grid(True)

print("   -> 正在反演连通系统 1D T2谱 (分别反演)...")
spec_c_tot = smart_invert_lcurve(times, amp_c_tot, t2_axis, inversion_mode, user_alpha) / norm_factor
spec_c_l = smart_invert_lcurve(times, amp_c_l, t2_axis, inversion_mode, user_alpha) / norm_factor
spec_c_s = smart_invert_lcurve(times, amp_c_s, t2_axis, inversion_mode, user_alpha) / norm_factor

axs2[2].fill_between(t2_axis, spec_c_l, color='#78C296', alpha=0.7, label='Large')
axs2[2].fill_between(t2_axis, spec_c_s, color='#E59866', alpha=0.7, label='Small')
axs2[2].plot(t2_axis, spec_c_tot, 'k-', lw=2, label='Total')
axs2[2].set_xscale('log');
axs2[2].set_xlim(1, 5000);
axs2[2].set_ylim(0, max(spec_c_tot) * 1.2)
axs2[2].set_title("1D T2 Spectrum (Coupled)")
axs2[2].legend();
axs2[2].grid(True, ls="--", alpha=0.5)

export_dict_decay['Coupled_Sw_100%_T2_Total'] = amp_c_tot / norm_factor
export_dict_t2['Coupled_Sw_100%_Total'] = spec_c_tot
export_dict_t2['Coupled_Sw_100%_Large'] = spec_c_l
export_dict_t2['Coupled_Sw_100%_Small'] = spec_c_s

print("   -> 正在执行 2D T2-T2 模拟 (运算量较大，请稍候)...")
t1_axis_2d = np.linspace(0, 1000, 15)
t2_axis_2d = np.linspace(0, 1500, 25)
t2_bins_2d = np.logspace(0, 3.5, 25)
tm_mix = 22.0

S_2d = solve_t2_t2_exchange(coupled_mesh, D_scaled, T2B_scaled, rho2_scaled, t1_axis_2d, tm_mix, t2_axis_2d)
F_2d = invert_t2_t2_nnls(S_2d, t1_axis_2d, t2_axis_2d, t2_bins_2d, alpha=1.0)

cf = axs2[3].contourf(t2_bins_2d, t2_bins_2d, F_2d.T, levels=20, cmap='hot_r')
axs2[3].plot([1, 10000], [1, 10000], 'k--', alpha=0.5)
axs2[3].set_xscale('log');
axs2[3].set_yscale('log')
axs2[3].set_xlim(10, 3000);
axs2[3].set_ylim(10, 3000)
axs2[3].set_xlabel('T2_2 (ms)');
axs2[3].set_ylabel('T2_1 (ms)')
axs2[3].set_title(f"T2-T2 Map (tm={tm_mix}ms)")

plt.tight_layout()
plt.show()

# ==========================================
# 6. 导出数据到 Excel
# ==========================================
print("\n" + "=" * 50)
print("导出数据")
print("=" * 50)

df_t2 = pd.DataFrame(export_dict_t2)
t2_file_path = os.path.abspath("T2_Inversion_Detailed_2.5D.xlsx")
df_t2.to_excel(t2_file_path, index=False)

# [新增] 导出 T1
df_t1 = pd.DataFrame(export_dict_t1)
t1_file_path = os.path.abspath("T1_Inversion_Detailed_2.5D.xlsx")
df_t1.to_excel(t1_file_path, index=False)

df_decay = pd.DataFrame(export_dict_decay)
decay_file_path = os.path.abspath("Triangle_Raw_Decay.xlsx")
df_decay.to_excel(decay_file_path, index=False)

df_t2_t2 = pd.DataFrame(F_2d.T, index=t2_bins_2d, columns=t2_bins_2d)
df_t2_t2.index.name = "T2_1 (ms) \ T2_2 (ms)"
t2_t2_file_path = os.path.abspath("T2_T2_Exchange_Map.xlsx")
df_t2_t2.to_excel(t2_t2_file_path)

print(f"--- Export Complete! ---")
print(f"T2 (1D) 结果已保存至: {t2_file_path}")
print(f"T1 (1D) 结果已保存至: {t1_file_path}")
print(f"衰减数据已保存至: {decay_file_path}")
print(f"T2-T2 (2D) 矩阵已保存至: {t2_t2_file_path}")