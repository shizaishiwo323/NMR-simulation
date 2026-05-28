import sys
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from scipy.optimize import nnls, brentq
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

from nmr_t2.pipelines import invert_single_signal_lcurve

output_dir = cfg_get("triangle_ca.output_dir")
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"Triangle CA output folder = {os.getcwd()}")

# ==========================================
# 0. 用户交互：选择反演模式与参数、设定接触角与内角
# ==========================================
print("=" * 50)
print("欢迎使用 NMR T2 滞后效应 (任意内角 + 防飞走版) 控制台")
print("=" * 50)

if cfg_get("run.interactive", True):
    theta_input = input("请输入接触角 Contact Angle (度, 例如 0, 30。直接回车默认 0): ").strip()
    theta_deg = float(theta_input) if theta_input else 0.0
else:
    theta_deg = float(cfg_get("triangle_ca.contact_angle_deg", cfg_get("geometry.contact_angle_deg", 0.0)))
    print(f"控制文件设定接触角: {theta_deg}°")

print("\n请设定三角形的三个内角 (以度为单位, 总和必须为180):")
if cfg_get("run.interactive", True):
    gamma1_in = input("角1 (默认 60): ").strip()
    gamma2_in = input("角2 (默认 60): ").strip()
    gamma3_in = input("角3 (默认 60): ").strip()
    g1 = float(gamma1_in) if gamma1_in else 60.0
    g2 = float(gamma2_in) if gamma2_in else 60.0
    g3 = float(gamma3_in) if gamma3_in else 60.0
else:
    g1, g2, g3 = [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    print(f"控制文件设定三角形内角: {g1}, {g2}, {g3}")

if abs(g1 + g2 + g3 - 180.0) > 1e-3:
    print("⚠️ 警告：三个角之和不为 180°！强制重置为等边三角形 (60, 60, 60)。")
    g1, g2, g3 = 60.0, 60.0, 60.0

gammas = [g1, g2, g3]

for g in gammas:
    if g / 2.0 + theta_deg >= 90.0:
        print(f"⚠️ 警告：接触角 {theta_deg}° 过大，角隅 {g}° 无法稳定持水。强制将接触角重置为 0°。")
        theta_deg = 0.0
        break

print("\n[1] 固定平滑参数模式 (适合无理想模拟数据)")
print("[2] L-Curve 自适应模式 (适合带噪声数据)")
if cfg_get("run.interactive", True):
    inversion_mode = input("请输入您的选择 (1 或 2): ").strip()
else:
    control_mode = str(cfg_get("inversion.mode", "fixed")).lower()
    inversion_mode = "2" if control_mode in {"2", "l_curve", "lcurve", "auto"} else "1"
    print(f"控制文件设定反演模式: {control_mode} -> {inversion_mode}")
if inversion_mode not in ['1', '2']: inversion_mode = '1'

user_alpha = 0.2
if inversion_mode == '1':
    if cfg_get("run.interactive", True):
        alpha_input = input("请输入固定平滑参数 alpha 的值 (直接回车默认 0.2): ").strip()
        if alpha_input: user_alpha = float(alpha_input)
    else:
        user_alpha = float(cfg_get("inversion.alpha", 0.2))
    print(f"已设定固定平滑参数: {user_alpha}")

# ==========================================
# 1. 物理参数及任意三角形几何计算
# ==========================================
D_scaled, T2B_scaled, rho_scaled = 2.0, 3000.0, 0.005
dt = 2.0
times = np.arange(0, 1500, dt)
t2_axis = np.logspace(0, 4, 150)

L_large, L_small = 20.0, 8.0
depth_large, depth_small = 20.0, 8.0


def calc_arbitrary_pore_properties(L_base, gammas_deg):
    """基于正弦定理计算任意三角形的全部几何及物理属性 (修复飞走Bug版)"""
    g_rad = [math.radians(g) for g in gammas_deg]

    c = L_base
    a = c * math.sin(g_rad[0]) / math.sin(g_rad[2])
    b = c * math.sin(g_rad[1]) / math.sin(g_rad[2])

    P = a + b + c
    A = 0.5 * b * c * math.sin(g_rad[0])
    G = A / (P ** 2)

    r_I = 2 * A / P
    r_D = P / (1.0 / (2.0 * G) + math.sqrt(math.pi / G))

    # 【终极防飞走方案】：底边严格居中法
    # 强制让三角形的底边关于 y 轴对称 (一半在左，一半在右)
    # 这样无论是大孔隙还是小孔隙，它们的底边必然在 x=0 处完美咬合，绝不分离！
    V1 = [-c / 2.0, 0.0]
    V2 = [c / 2.0, 0.0]
    V3 = [b * math.cos(g_rad[0]) - c / 2.0, b * math.sin(g_rad[0])]

    # 内心 (Incenter) 坐标 - 所有角平分线的交点
    Ix = (a * V1[0] + b * V2[0] + c * V3[0]) / P
    Iy = (a * V1[1] + b * V2[1] + c * V3[1]) / P

    return A, P, r_I, r_D, [V1, V2, V3], [Ix, Iy]


A_l, P_l, r_I_l, r_D_l, verts_l, inc_l = calc_arbitrary_pore_properties(L_large, gammas)
A_s, P_s, r_I_s, r_D_s, verts_s, inc_s = calc_arbitrary_pore_properties(L_small, gammas)

V_large = A_l * depth_large
V_small = A_s * depth_small
total_pore_volume = V_large + V_small


# ==========================================
# 2. 任意三角形角隅水计算 (独立分配三个角)
# ==========================================
def calc_corner_water_area_arb(r_m, gammas_deg, theta_d):
    theta_rad = math.radians(theta_d)
    total_area = 0.0
    F_factors = []

    for g in gammas_deg:
        alpha_rad = math.radians(g) / 2.0
        F = (math.cos(theta_rad) * math.cos(alpha_rad + theta_rad) / math.sin(alpha_rad)) - \
            (math.pi / 2.0 - alpha_rad - theta_rad)
        F_factors.append(F)
        if F > 0:
            total_area += (r_m ** 2) * F

    return total_area, F_factors


def get_single_pore_water_area(L, r_m, process, theta_d, gammas_deg):
    A, P, r_I, r_D, _, _ = calc_arbitrary_pore_properties(L, gammas_deg)
    if process == 'Drainage':
        if r_m >= r_D: return A
        cw_area, _ = calc_corner_water_area_arb(r_m, gammas_deg, theta_d)
        return min(A, cw_area)
    elif process == 'Imbibition':
        if r_m >= r_I: return A
        cw_area, _ = calc_corner_water_area_arb(r_m, gammas_deg, theta_d)
        return min(A, cw_area)


def get_global_Sw(r_m, process, theta_d, gammas_deg):
    Aw_l = get_single_pore_water_area(L_large, r_m, process, theta_d, gammas_deg)
    Aw_s = get_single_pore_water_area(L_small, r_m, process, theta_d, gammas_deg)
    return (Aw_l * depth_large + Aw_s * depth_small) / total_pore_volume


# ==========================================
# 3. 不对称网格剖分引擎
# ==========================================
def create_pore_with_bubble_arb(A_tot, vertices, incenter, gammas_deg, target_air_area, theta_d=0.0, is_inverted=False):
    if is_inverted:
        verts = [[v[0], -v[1]] for v in vertices]
        inc = [incenter[0], -incenter[1]]
    else:
        verts, inc = vertices, incenter

    A_w = A_tot - target_air_area
    if A_w < 1e-5: return pg.Mesh(2)

    if target_air_area < 1e-4:
        geom = mt.createPolygon(verts, isClosed=True)
        for b in geom.boundaries(): b.setMarker(1)
        return mt.createMesh(geom, area=A_tot / 800.0, quality=33)

    theta_rad = math.radians(theta_d)
    total_F_area, F_factors = calc_corner_water_area_arb(1.0, gammas_deg, theta_d)
    R_m = math.sqrt(A_w / total_F_area)

    def unit(v):
        n = math.sqrt(v[0] ** 2 + v[1] ** 2); return [v[0] / n, v[1] / n]

    polys = []
    for i in range(3):
        if F_factors[i] <= 0: continue

        V_corner = verts[i]
        V_adj1 = verts[(i + 1) % 3]
        V_adj2 = verts[(i - 1) % 3]
        alpha_rad = math.radians(gammas_deg[i]) / 2.0
        L_c = R_m * math.cos(alpha_rad + theta_rad) / math.sin(alpha_rad)
        d_c = R_m * math.cos(theta_rad) / math.sin(alpha_rad)

        dir1 = unit([V_adj1[0] - V_corner[0], V_adj1[1] - V_corner[1]])
        dir2 = unit([V_adj2[0] - V_corner[0], V_adj2[1] - V_corner[1]])
        dir_bisect = unit([inc[0] - V_corner[0], inc[1] - V_corner[1]])
        C_arc = [V_corner[0] + d_c * dir_bisect[0], V_corner[1] + d_c * dir_bisect[1]]

        P1 = [V_corner[0] + L_c * dir1[0], V_corner[1] + L_c * dir1[1]]
        P2 = [V_corner[0] + L_c * dir2[0], V_corner[1] + L_c * dir2[1]]

        angle1, angle2 = math.atan2(P1[1] - C_arc[1], P1[0] - C_arc[0]), math.atan2(P2[1] - C_arc[1], P2[0] - C_arc[0])
        if angle2 - angle1 > math.pi:
            angle2 -= 2 * math.pi
        elif angle1 - angle2 > math.pi:
            angle2 += 2 * math.pi

        arc_pts = [[C_arc[0] + R_m * math.cos(angle1 + (angle2 - angle1) * j / 29),
                    C_arc[1] + R_m * math.sin(angle1 + (angle2 - angle1) * j / 29)] for j in range(30)]

        poly = mt.createPolygon([V_corner] + arc_pts, isClosed=True)
        for b in poly.boundaries():
            bcen = b.center()
            dist = math.sqrt((bcen[0] - C_arc[0]) ** 2 + (bcen[1] - C_arc[1]) ** 2)
            b.setMarker(99 if abs(dist - R_m) < R_m * 0.05 else 1)
        polys.append(poly)

    final_geom = polys[0]
    for p in polys[1:]: final_geom += p
    return mt.createMesh(final_geom, area=A_tot / 800.0, quality=33)


def solve_single_decay(mesh, D, T2B, rho, times, dt):
    if mesh.nodeCount() < 5: return np.zeros(len(times))
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    amps, a, b = [], D * dt, -(1.0 + dt / T2B)
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


def invert_t2_fixed(times, signal, t2_bins, alpha):
    if np.max(signal) <= 1e-5: return np.zeros(len(t2_bins))
    A = np.exp(-np.outer(times, 1.0 / t2_bins))
    n = len(t2_bins)
    L = np.zeros((n, n))
    [L.__setitem__((i, i - 1), 1) or L.__setitem__((i, i), -2) or L.__setitem__((i, i + 1), 1) for i in range(1, n - 1)]
    L[0, 0] = -1;
    L[0, 1] = 1;
    L[-1, -2] = 1;
    L[-1, -1] = -1
    spectrum, _ = nnls(np.vstack((A, L * alpha)), np.concatenate((signal / signal[0], np.zeros(n))))
    return spectrum


def smart_invert_lcurve(times, signal, t2_axis, mode, current_alpha):
    if np.max(signal) <= 1e-5: return np.zeros(len(t2_axis))
    max_amp = np.max(signal)
    if mode == '1':
        return invert_t2_fixed(times, signal, t2_axis, current_alpha) * max_amp
    else:
        norm_signal = signal / max_amp
        try:
            result = invert_single_signal_lcurve(times, norm_signal + np.random.normal(0, 0.01, len(norm_signal)),
                                                 signal_name="Sim")
            spec = np.asarray(result.spectrum) if hasattr(result, 'spectrum') else np.asarray(result)
            if len(spec) != len(t2_axis): spec = np.interp(t2_axis, np.logspace(0, 4, len(spec)), spec)
            return spec * max_amp
        except Exception:
            return invert_t2_fixed(times, signal, t2_axis, current_alpha) * max_amp


# ==========================================
# 4. 全景图输出设计
# ==========================================
selected_rm = [r_I_l * 1.5, r_I_l * 0.99, r_D_l * 0.99, r_I_s * 0.99, r_D_s * 0.99]
processes = list(cfg_get("triangle_ca.processes", ['Drainage', 'Imbibition']))
num_cols = len(selected_rm)

fig_wrc, ax_wrc = plt.subplots(figsize=(10, 6))
fig_wrc.suptitle(f"Page 1: Water Retention Curve ({gammas[0]}°, {gammas[1]}°, {gammas[2]}°)", fontsize=18,
                 fontweight='bold')

rm_array = np.logspace(-0.5, 1.5, 300)
pc_array = 1.0 / rm_array
sw_d_array = [get_global_Sw(r, 'Drainage', theta_deg, gammas) for r in rm_array]
sw_i_array = [get_global_Sw(r, 'Imbibition', theta_deg, gammas) for r in rm_array]

ax_wrc.semilogx(pc_array, sw_d_array, 'b-', lw=2, label='Drainage (Desaturation)')
ax_wrc.semilogx(pc_array, sw_i_array, 'r-', lw=2, label='Imbibition (Resaturation)')

selected_pc = [1.0 / r for r in selected_rm]
sw_d_selected = [get_global_Sw(r, 'Drainage', theta_deg, gammas) for r in selected_rm]
sw_i_selected = [get_global_Sw(r, 'Imbibition', theta_deg, gammas) for r in selected_rm]

ax_wrc.plot(selected_pc, sw_d_selected, 'bo', markersize=8)
ax_wrc.plot(selected_pc, sw_i_selected, 'ro', markersize=8)

for i, pc in enumerate(selected_pc):
    ax_wrc.axvline(pc, color='gray', linestyle='--', alpha=0.5)
    ax_wrc.text(pc, 1.02, f"P{i + 1}", ha='center', fontsize=12, fontweight='bold')

ax_wrc.set_xlabel("Capillary Pressure ~ 1/rm [1/μm]", fontsize=14)
ax_wrc.set_ylabel("Global Saturation Sw [-]", fontsize=14)
ax_wrc.legend(fontsize=12)
ax_wrc.grid(True, which="both", ls="--", alpha=0.4)
fig_wrc.tight_layout()

# ----------------- Page 2, 3, 4 -----------------
fig_morph, axs_morph = plt.subplots(2, num_cols, figsize=(4.5 * num_cols, 12), squeeze=False)
fig_morph.suptitle("Page 2: Arbitrary Pore Morphologies", fontsize=20, fontweight='bold')

fig_decay, axs_decay = plt.subplots(2, num_cols, figsize=(4.5 * num_cols, 10), squeeze=False)
fig_decay.suptitle("Page 3: Decay Curves", fontsize=20, fontweight='bold')

fig_t2, axs_t2 = plt.subplots(2, num_cols, figsize=(4.5 * num_cols, 10), squeeze=False)
fig_t2.suptitle("Page 4: T2 Inversion Spectra", fontsize=20, fontweight='bold')

export_dict_t2 = {'T2_Time_ms': t2_axis}
export_dict_decay = {'Time': times}

for r_idx, process in enumerate(processes):
    for c_idx, rm_val in enumerate(selected_rm):
        actual_sw = get_global_Sw(rm_val, process, theta_deg, gammas)
        title_str = f"P{c_idx + 1}: Sw={actual_sw * 100:.1f}%"
        print(f"[{process}] 计算点 P{c_idx + 1} | 真实 Sw = {actual_sw * 100:.1f}%")

        Aw_l = get_single_pore_water_area(L_large, rm_val, process, theta_deg, gammas)
        Aw_s = get_single_pore_water_area(L_small, rm_val, process, theta_deg, gammas)

        mesh_l = create_pore_with_bubble_arb(A_l, verts_l, inc_l, gammas, A_l - Aw_l, theta_deg, False)
        mesh_s = create_pore_with_bubble_arb(A_s, verts_s, inc_s, gammas, A_s - Aw_s, theta_deg, True)

        # 1. 画形态
        ax_m = axs_morph[r_idx, c_idx]

        # 绘制黑线边框
        v_l_draw = verts_l + [verts_l[0]]
        ax_m.plot([v[0] for v in v_l_draw], [v[1] for v in v_l_draw], 'k-', lw=1.2)
        v_s_draw = [[v[0], -v[1]] for v in verts_s] + [[verts_s[0][0], -verts_s[0][1]]]
        ax_m.plot([v[0] for v in v_s_draw], [v[1] for v in v_s_draw], 'k-', lw=1.2)

        if mesh_l.nodeCount() > 0: pg.show(mesh_l, ax=ax_m, hold=True)
        if mesh_s.nodeCount() > 0: pg.show(mesh_s, ax=ax_m, hold=True)
        ax_m.set_title(title_str, fontsize=14);
        ax_m.set_aspect('equal')

        # 【动态视场捕捉 (Dynamic Camera)】：防止极端倾斜角导致三角形出界
        all_x = [v[0] for v in verts_l] + [v[0] for v in verts_s]
        all_y = [v[1] for v in verts_l] + [-v[1] for v in verts_s]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        margin_x = (max_x - min_x) * 0.15 + 2
        margin_y = (max_y - min_y) * 0.15 + 2
        ax_m.set_xlim(min_x - margin_x, max_x + margin_x)
        ax_m.set_ylim(min_y - margin_y, max_y + margin_y)

        # 2. 画衰减
        amp_l_vol = solve_single_decay(mesh_l, D_scaled, T2B_scaled, rho_scaled, times, dt) * depth_large
        amp_s_vol = solve_single_decay(mesh_s, D_scaled, T2B_scaled, rho_scaled, times, dt) * depth_small
        at_vol = amp_l_vol + amp_s_vol
        norm_l_vol, norm_s_vol, norm_at_vol = amp_l_vol / total_pore_volume, amp_s_vol / total_pore_volume, at_vol / total_pore_volume

        ax_d = axs_decay[r_idx, c_idx]
        ax_d.plot(times, norm_l_vol, color='#78C296', label='Large')
        ax_d.plot(times, norm_s_vol, color='#E59866', label='Small')
        ax_d.plot(times, norm_at_vol, 'k--', label='Total')
        ax_d.set_ylim(0, 1.1);
        ax_d.set_title(title_str, fontsize=14);
        ax_d.grid(True, alpha=0.3)
        if c_idx == 0: ax_d.legend(loc='upper right')

        if c_idx >= 2:
            axins_d = ax_d.inset_axes([0.35, 0.40, 0.60, 0.50])
            axins_d.plot(times, norm_l_vol, color='#78C296')
            axins_d.plot(times, norm_s_vol, color='#E59866')
            axins_d.plot(times, norm_at_vol, 'k--')
            max_d = np.max(norm_at_vol) if len(norm_at_vol) > 0 else 0
            axins_d.set_ylim(0, max_d * 1.3 if max_d > 1e-5 else 1e-4)
            axins_d.grid(True, alpha=0.3);
            axins_d.tick_params(labelsize=8)

        # 3. 画 T2
        spec_l = smart_invert_lcurve(times, amp_l_vol, t2_axis, inversion_mode, user_alpha) / total_pore_volume
        spec_s = smart_invert_lcurve(times, amp_s_vol, t2_axis, inversion_mode, user_alpha) / total_pore_volume
        total_spec = spec_l + spec_s

        ax_t = axs_t2[r_idx, c_idx]
        ax_t.fill_between(t2_axis, spec_l, color='#78C296', alpha=0.7)
        ax_t.fill_between(t2_axis, spec_s, color='#E59866', alpha=0.7)
        ax_t.plot(t2_axis, total_spec, 'k-')
        ax_t.set_xscale('log');
        ax_t.set_xlim(1, 5000);
        ax_t.set_ylim(0, 0.4)
        ax_t.set_title(title_str, fontsize=14);
        ax_t.grid(True, which="both", ls="--", alpha=0.3)

        if c_idx >= 2:
            axins_t = ax_t.inset_axes([0.35, 0.45, 0.60, 0.45])
            axins_t.fill_between(t2_axis, spec_l, color='#78C296', alpha=0.7)
            axins_t.fill_between(t2_axis, spec_s, color='#E59866', alpha=0.7)
            axins_t.plot(t2_axis, total_spec, 'k-')
            axins_t.set_xscale('log');
            axins_t.set_xlim(1, 5000)
            max_t = np.max(total_spec) if len(total_spec) > 0 else 0
            axins_t.set_ylim(0, max_t * 1.3 if max_t > 1e-5 else 1e-4)
            axins_t.grid(True, which="both", ls="--", alpha=0.3);
            axins_t.tick_params(labelsize=8)

        # 4. 存数据
        col_prefix = f'P{c_idx + 1}_{process}'
        export_dict_t2[f'{col_prefix}_Small'] = spec_s
        export_dict_t2[f'{col_prefix}_Large'] = spec_l
        export_dict_t2[f'{col_prefix}_Total'] = total_spec
        export_dict_decay[f'{col_prefix}_Peak'] = norm_at_vol

fig_morph.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
fig_decay.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
fig_t2.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)

fig_wrc.savefig("Figure_1_Water_Retention_Curve.png", dpi=300, bbox_inches="tight")
fig_morph.savefig("Figure_2_Morphologies.png", dpi=300, bbox_inches="tight")
fig_decay.savefig("Figure_3_Decay_Curves.png", dpi=300, bbox_inches="tight")
fig_t2.savefig("Figure_4_T2_Spectra.png", dpi=300, bbox_inches="tight")

if bool(cfg_get("run.show_figures", True)):
    plt.show()
else:
    print("控制文件设置 show_figures=False；不弹出图窗，只保存图片。")
    plt.close("all")
# 111
# ==========================================
# 5. 导出数据到 Excel
# ==========================================
print("\n正在导出全过程数据到 Excel ...")
df_t2 = pd.DataFrame(export_dict_t2)
t2_file_path = os.path.abspath(f"T2_ArbTri_Inversion.xlsx")
df_t2.to_excel(t2_file_path, index=False)

df_decay = pd.DataFrame(export_dict_decay)
decay_file_path = os.path.abspath(f"T2_ArbTri_Decay.xlsx")
df_decay.to_excel(decay_file_path, index=False)

with open("Simulation_Control_Used.json", "w", encoding="utf-8") as f:
    json.dump(SIMULATION_CONTROL, f, indent=2, ensure_ascii=False)

print(f"--- 导出完成！已保存为 T2_ArbTri 系列 ---")
