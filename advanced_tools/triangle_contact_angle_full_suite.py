"""Full contact-angle triangle suite: T2, T2-T2, and D-T2 for all selected states."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygimli as pg
import pygimli.meshtools as mt
from scipy.optimize import nnls

from simulation_control import SETTINGS


def cfg_get(path, default=None):
    value = SETTINGS
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


D_scaled, T2B_scaled, rho_scaled = 2.0, 3000.0, 0.005
dt = 2.0
times = np.arange(0.0, 1500.0, dt)
t2_axis = np.logspace(0.0, 4.0, 150)

L_large, L_small = 20.0, 8.0
depth_large, depth_small = 20.0, 8.0
L_t, W_t = 5.0, 1.0


def safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("%", "pct")
        .replace("°", "deg")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def calc_arbitrary_pore_properties(L_base, gammas_deg):
    g_rad = [math.radians(g) for g in gammas_deg]
    c = L_base
    a = c * math.sin(g_rad[0]) / math.sin(g_rad[2])
    b = c * math.sin(g_rad[1]) / math.sin(g_rad[2])
    perimeter = a + b + c
    area = 0.5 * b * c * math.sin(g_rad[0])
    shape_factor = area / (perimeter**2)
    r_i = 2.0 * area / perimeter
    r_d = perimeter / (1.0 / (2.0 * shape_factor) + math.sqrt(math.pi / shape_factor))
    v1 = [-c / 2.0, 0.0]
    v2 = [c / 2.0, 0.0]
    v3 = [b * math.cos(g_rad[0]) - c / 2.0, b * math.sin(g_rad[0])]
    ix = (a * v1[0] + b * v2[0] + c * v3[0]) / perimeter
    iy = (a * v1[1] + b * v2[1] + c * v3[1]) / perimeter
    return area, perimeter, r_i, r_d, [v1, v2, v3], [ix, iy]


def calc_corner_water_area_arb(r_m, gammas_deg, theta_d):
    theta_rad = math.radians(theta_d)
    total_area = 0.0
    factors = []
    for gamma in gammas_deg:
        alpha_rad = math.radians(gamma) / 2.0
        factor = (
            math.cos(theta_rad) * math.cos(alpha_rad + theta_rad) / math.sin(alpha_rad)
            - (math.pi / 2.0 - alpha_rad - theta_rad)
        )
        factors.append(factor)
        if factor > 0.0:
            total_area += r_m**2 * factor
    return total_area, factors


def get_single_pore_water_area(L, r_m, process, theta_d, gammas_deg):
    area, _, r_i, r_d, _, _ = calc_arbitrary_pore_properties(L, gammas_deg)
    if process == "Drainage":
        if r_m >= r_d:
            return area
    elif process == "Imbibition":
        if r_m >= r_i:
            return area
    else:
        raise ValueError(f"Unknown process: {process}")
    corner_area, _ = calc_corner_water_area_arb(r_m, gammas_deg, theta_d)
    return min(area, corner_area)


def make_polygon_geom(points, marker=1):
    geom = mt.createPolygon(points, isClosed=True)
    for boundary in geom.boundaries():
        boundary.setMarker(marker)
    return geom


def transform_points(points, *, invert=False, y_shift=0.0):
    if invert:
        return [[p[0], -p[1] + y_shift] for p in points]
    return [[p[0], p[1] + y_shift] for p in points]


def create_pore_geom_arb(area_total, vertices, incenter, gammas_deg, target_air_area, theta_d=0.0, is_inverted=False, y_shift=0.0):
    verts = transform_points(vertices, invert=is_inverted, y_shift=y_shift)
    inc = transform_points([incenter], invert=is_inverted, y_shift=y_shift)[0]
    water_area = area_total - target_air_area
    if water_area < 1e-5:
        return None, False
    if target_air_area < 1e-4:
        return make_polygon_geom(verts, marker=1), True

    theta_rad = math.radians(theta_d)
    unit_area, factors = calc_corner_water_area_arb(1.0, gammas_deg, theta_d)
    if unit_area <= 0.0:
        return None, False
    r_m = math.sqrt(water_area / unit_area)

    def unit(v):
        norm = math.hypot(v[0], v[1])
        return [v[0] / norm, v[1] / norm]

    polygons = []
    for i in range(3):
        if factors[i] <= 0.0:
            continue
        v_corner = verts[i]
        v_adj1 = verts[(i + 1) % 3]
        v_adj2 = verts[(i - 1) % 3]
        alpha_rad = math.radians(gammas_deg[i]) / 2.0
        l_c = r_m * math.cos(alpha_rad + theta_rad) / math.sin(alpha_rad)
        d_c = r_m * math.cos(theta_rad) / math.sin(alpha_rad)
        dir1 = unit([v_adj1[0] - v_corner[0], v_adj1[1] - v_corner[1]])
        dir2 = unit([v_adj2[0] - v_corner[0], v_adj2[1] - v_corner[1]])
        dir_bisect = unit([inc[0] - v_corner[0], inc[1] - v_corner[1]])
        c_arc = [v_corner[0] + d_c * dir_bisect[0], v_corner[1] + d_c * dir_bisect[1]]
        p1 = [v_corner[0] + l_c * dir1[0], v_corner[1] + l_c * dir1[1]]
        p2 = [v_corner[0] + l_c * dir2[0], v_corner[1] + l_c * dir2[1]]
        angle1 = math.atan2(p1[1] - c_arc[1], p1[0] - c_arc[0])
        angle2 = math.atan2(p2[1] - c_arc[1], p2[0] - c_arc[0])
        if angle2 - angle1 > math.pi:
            angle2 -= 2.0 * math.pi
        elif angle1 - angle2 > math.pi:
            angle2 += 2.0 * math.pi
        arc = [
            [c_arc[0] + r_m * math.cos(angle1 + (angle2 - angle1) * j / 29.0),
             c_arc[1] + r_m * math.sin(angle1 + (angle2 - angle1) * j / 29.0)]
            for j in range(30)
        ]
        geom = mt.createPolygon([v_corner] + arc, isClosed=True)
        for boundary in geom.boundaries():
            center = boundary.center()
            dist = math.hypot(center.x() - c_arc[0], center.y() - c_arc[1])
            boundary.setMarker(99 if abs(dist - r_m) < r_m * 0.05 else 1)
        polygons.append(geom)

    if not polygons:
        return None, False
    final = polygons[0]
    for polygon in polygons[1:]:
        final += polygon
    return final, False


def create_mesh_from_geom(geom, area_total, fine_factor=900.0):
    if geom is None:
        return pg.Mesh(2)
    return mt.createMesh(geom, area=max(area_total / fine_factor, 0.02), quality=33)


def solve_single_decay(mesh, D, T_bulk, rho, times_axis, step_dt):
    if mesh.nodeCount() < 5:
        return np.zeros(len(times_axis))
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    amps = []
    a, b = D * step_dt, -(1.0 + step_dt / T_bulk)
    bc = {"Robin": {1: rho * step_dt}}
    for _ in times_axis:
        u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
        amps.append(np.sum(u_cell * np.asarray(mesh.cellSizes())))
        try:
            u = np.asarray(pg.solve(mesh, a=a, b=b, f=np.asarray(u, dtype=np.float64), bc=bc), dtype=np.float64)
        except Exception:
            u = np.asarray(pg.solver.solve(mesh, a=a, b=b, f=np.asarray(u, dtype=np.float64), bc=bc), dtype=np.float64)
    return np.asarray(amps)


def build_coupled_cell_volumes(mesh):
    cell_areas = np.asarray(mesh.cellSizes())
    centers_y = np.array([cell.center().y() for cell in mesh.cells()])
    volumes = np.zeros_like(cell_areas)
    idx_large = centers_y > L_t / 2.0
    idx_small = centers_y < -L_t / 2.0
    volumes[idx_large] = cell_areas[idx_large] * depth_large
    volumes[idx_small] = cell_areas[idx_small] * depth_small
    return volumes, idx_large, idx_small


def solve_coupled_decay(mesh, times_axis, step_dt):
    if mesh.nodeCount() < 5:
        return np.zeros(len(times_axis)), np.zeros(len(times_axis)), np.zeros(len(times_axis))
    u = np.ones(mesh.nodeCount(), dtype=np.float64)
    volumes, idx_large, idx_small = build_coupled_cell_volumes(mesh)
    amps_total, amps_large, amps_small = [], [], []
    a, b = D_scaled * step_dt, -(1.0 + step_dt / T2B_scaled)
    bc = {"Robin": {1: rho_scaled * step_dt}}
    for _ in times_axis:
        u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
        amps_total.append(np.sum(u_cell * volumes))
        amps_large.append(np.sum(u_cell[idx_large] * volumes[idx_large]))
        amps_small.append(np.sum(u_cell[idx_small] * volumes[idx_small]))
        try:
            u = np.asarray(pg.solve(mesh, a=a, b=b, f=np.asarray(u, dtype=np.float64), bc=bc), dtype=np.float64)
        except Exception:
            u = np.asarray(pg.solver.solve(mesh, a=a, b=b, f=np.asarray(u, dtype=np.float64), bc=bc), dtype=np.float64)
    return np.asarray(amps_total), np.asarray(amps_large), np.asarray(amps_small)


def _pg_solve_step(mesh, field, step_dt):
    a, b = D_scaled * step_dt, -(1.0 + step_dt / T2B_scaled)
    bc = {"Robin": {1: rho_scaled * step_dt}}
    try:
        return np.asarray(pg.solve(mesh, a=a, b=b, f=np.asarray(field, dtype=np.float64), bc=bc), dtype=np.float64)
    except Exception:
        return np.asarray(pg.solver.solve(mesh, a=a, b=b, f=np.asarray(field, dtype=np.float64), bc=bc), dtype=np.float64)


def invert_t2_fixed(times_axis, signal, bins, alpha):
    if np.max(signal) <= 1e-12:
        return np.zeros(len(bins))
    kernel = np.exp(-np.outer(times_axis, 1.0 / bins))
    n = len(bins)
    reg = np.zeros((n, n))
    for i in range(1, n - 1):
        reg[i, i - 1] = 1.0
        reg[i, i] = -2.0
        reg[i, i + 1] = 1.0
    reg[0, 0] = -1.0
    reg[0, 1] = 1.0
    reg[-1, -2] = 1.0
    reg[-1, -1] = -1.0
    norm = signal / max(signal[0], 1e-12)
    spec, _ = nnls(np.vstack((kernel, reg * alpha)), np.concatenate((norm, np.zeros(n))))
    return spec * signal[0]


def make_second_derivative_matrix(n):
    reg = np.zeros((n, n))
    for i in range(1, n - 1):
        reg[i, i - 1] = 1.0
        reg[i, i] = -2.0
        reg[i, i + 1] = 1.0
    reg[0, 0] = -1.0
    reg[0, 1] = 1.0
    reg[-1, -2] = 1.0
    reg[-1, -1] = -1.0
    return reg


def solve_t2_t2_exchange(mesh, times1, tm, times2):
    volumes, _, _ = build_coupled_cell_volumes(mesh)
    signal = np.zeros((len(times1), len(times2)))
    for i, t1 in enumerate(times1):
        u = np.ones(mesh.nodeCount(), dtype=np.float64)
        for duration in [t1, tm]:
            steps = max(1, int(duration / 2.0)) if duration > 0 else 0
            if steps == 0:
                continue
            step_dt = duration / steps
            a, b = D_scaled * step_dt, -(1.0 + step_dt / T2B_scaled)
            bc = {"Robin": {1: rho_scaled * step_dt}}
            for _ in range(steps):
                u = np.asarray(pg.solve(mesh, a=a, b=b, f=np.asarray(u, dtype=np.float64), bc=bc), dtype=np.float64)
        for j, t2 in enumerate(times2):
            u_cell = np.asarray(pg.interpolate(mesh, u, mesh.cellCenters()))
            signal[i, j] = np.sum(u_cell * volumes)
            if j < len(times2) - 1:
                step_dt = times2[j + 1] - times2[j]
                a, b = D_scaled * step_dt, -(1.0 + step_dt / T2B_scaled)
                bc = {"Robin": {1: rho_scaled * step_dt}}
                u = np.asarray(pg.solve(mesh, a=a, b=b, f=np.asarray(u, dtype=np.float64), bc=bc), dtype=np.float64)
    return signal


def invert_t2_t2_nnls(signal, times1, times2, bins, alpha=0.05):
    a1 = np.exp(-np.outer(times1, 1.0 / bins))
    a2 = np.exp(-np.outer(times2, 1.0 / bins))
    kernel = np.kron(a2, a1)
    svec = signal.flatten("F")
    n = len(bins)
    reg1 = np.zeros((n, n))
    for i in range(1, n - 1):
        reg1[i, i - 1] = 1.0
        reg1[i, i] = -2.0
        reg1[i, i + 1] = 1.0
    reg = np.kron(np.eye(n), reg1) + np.kron(reg1, np.eye(n))
    fvec, _ = nnls(np.vstack((kernel, alpha * reg)), np.concatenate((svec, np.zeros(reg.shape[0]))))
    return fvec.reshape((n, n), order="F")


def mesh_node_coordinate(mesh, axis="x"):
    if axis.lower() == "x":
        return np.array([node.pos().x() for node in mesh.nodes()], dtype=np.float64)
    if axis.lower() == "y":
        return np.array([node.pos().y() for node in mesh.nodes()], dtype=np.float64)
    raise ValueError("axis must be 'x' or 'y'")


def evolve_complex_field(mesh, u_complex, duration, max_dt):
    if duration <= 1e-12:
        return u_complex
    steps = max(1, int(math.ceil(duration / max_dt)))
    step_dt = duration / steps
    ur = np.asarray(np.real(u_complex), dtype=np.float64)
    ui = np.asarray(np.imag(u_complex), dtype=np.float64)
    for _ in range(steps):
        ur = _pg_solve_step(mesh, ur, step_dt)
        ui = _pg_solve_step(mesh, ui, step_dt)
    return ur + 1j * ui


def integrate_complex_over_cells(mesh, u_complex, cell_volumes):
    centers = mesh.cellCenters()
    ur_cell = np.asarray(pg.interpolate(mesh, np.real(u_complex), centers))
    ui_cell = np.asarray(pg.interpolate(mesh, np.imag(u_complex), centers))
    return np.sum((ur_cell + 1j * ui_cell) * cell_volumes)


def solve_dt2_signal_pfg(mesh, cell_volumes, te_axis, b_axis, dt_pde, delta_diff, grad_axis, progress_name=""):
    te_axis = np.asarray(te_axis, dtype=np.float64)
    b_axis = np.asarray(b_axis, dtype=np.float64)
    if np.min(te_axis) < delta_diff:
        raise ValueError("Minimum TE must be >= delta_diff for D-T2 PFG simulation.")
    coord = mesh_node_coordinate(mesh, axis=grad_axis)
    signal = np.zeros((len(b_axis), len(te_axis)), dtype=np.float64)
    for ib, b_value in enumerate(b_axis):
        if progress_name:
            print(f"      {progress_name}: b [{ib + 1}/{len(b_axis)}] = {b_value:.4g}")
        q = math.sqrt(max(float(b_value), 0.0) / max(delta_diff, 1e-12))
        phase_plus = np.exp(1j * q * coord)
        phase_minus = np.conj(phase_plus)
        u = evolve_complex_field(mesh, phase_plus.copy(), delta_diff, dt_pde)
        u *= phase_minus
        current_t = delta_diff
        for it, te in enumerate(te_axis):
            if te > current_t + 1e-12:
                u = evolve_complex_field(mesh, u, te - current_t, dt_pde)
                current_t = te
            signal[ib, it] = np.real(integrate_complex_over_cells(mesh, u, cell_volumes))
    return signal


def invert_d_t2_nnls(signal, b_axis, te_axis, d_bins, t2_bins, alpha=0.03):
    signal = np.asarray(signal, dtype=np.float64)
    if np.max(np.abs(signal)) <= 1e-12:
        return np.zeros((len(d_bins), len(t2_bins)))
    norm0 = signal[0, 0] if abs(signal[0, 0]) > 1e-12 else np.max(np.abs(signal))
    signal_norm = np.maximum(signal / norm0, 0.0)
    ad = np.exp(-np.outer(b_axis, d_bins))
    at = np.exp(-np.outer(te_axis, 1.0 / t2_bins))
    kernel = np.kron(at, ad)
    svec = signal_norm.flatten("F")
    n_d, n_t = len(d_bins), len(t2_bins)
    ld = make_second_derivative_matrix(n_d)
    lt = make_second_derivative_matrix(n_t)
    reg = np.vstack((np.kron(np.eye(n_t), ld), np.kron(lt, np.eye(n_d))))
    fvec, _ = nnls(np.vstack((kernel, alpha * reg)), np.concatenate((svec, np.zeros(reg.shape[0]))))
    return fvec.reshape((n_d, n_t), order="F")


def solve_uncoupled_dt2_signal(mesh_l, mesh_s, te_axis, b_axis, dt_pde, delta_diff, grad_axis):
    signal = np.zeros((len(b_axis), len(te_axis)), dtype=np.float64)
    if mesh_l.nodeCount() > 5:
        vols_l = np.asarray(mesh_l.cellSizes()) * depth_large
        signal += solve_dt2_signal_pfg(mesh_l, vols_l, te_axis, b_axis, dt_pde, delta_diff, grad_axis, "Uncoupled large")
    if mesh_s.nodeCount() > 5:
        vols_s = np.asarray(mesh_s.cellSizes()) * depth_small
        signal += solve_dt2_signal_pfg(mesh_s, vols_s, te_axis, b_axis, dt_pde, delta_diff, grad_axis, "Uncoupled small")
    return signal


def plot_t2_comparison(path, title, unc_spec, coup_spec):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(t2_axis, unc_spec, "k-", lw=2, label="Uncoupled")
    ax.plot(t2_axis, coup_spec, "r--", lw=2, label="Coupled")
    ax.set_xscale("log")
    ax.set_xlabel("T2 (ms)")
    ax.set_ylabel("T2 amplitude / total pore volume")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def split_coupled_spectrum(total_spec, large_fraction):
    large_fraction = float(np.clip(large_fraction, 0.0, 1.0))
    if np.max(total_spec) <= 1e-12:
        return np.zeros_like(total_spec), np.zeros_like(total_spec)
    log_t2 = np.log10(t2_axis)
    best_valley = t2_axis[0]
    min_diff = np.inf
    k_overlap = 5.0
    for valley in t2_axis:
        weight_small = 1.0 - (1.0 / (1.0 + np.exp(-k_overlap * (log_t2 - np.log10(valley)))))
        small_area = np.sum(total_spec * weight_small) / max(np.sum(total_spec), 1e-12)
        diff = abs((1.0 - small_area) - large_fraction)
        if diff < min_diff:
            min_diff = diff
            best_valley = valley
    weight_large = 1.0 / (1.0 + np.exp(-k_overlap * (log_t2 - np.log10(best_valley))))
    return total_spec * weight_large, total_spec * (1.0 - weight_large)


def plot_map(path, matrix, x_axis, y_axis, title, cmap="hot_r"):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    z = matrix
    if z.shape == (len(x_axis), len(y_axis)):
        z = z.T
    elif z.shape != (len(y_axis), len(x_axis)):
        raise ValueError(f"Map shape {z.shape} does not match axes x={len(x_axis)}, y={len(y_axis)}")
    ax.contourf(x_axis, y_axis, z, levels=25, cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T2 detect (ms)")
    ax.set_ylabel("T2 encode / D axis")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    gammas = [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    theta = float(cfg_get("triangle_ca.contact_angle_deg", 0.0))
    angle_token = "-".join(str(int(g)) if abs(g - int(g)) < 1e-9 else str(g).replace(".", "p") for g in gammas)
    theta_token = str(theta).replace(".", "p").replace("-", "m")
    out_dir = Path("simulation_outputs") / f"triangle_full_CA{theta_token}deg_angles_{angle_token}"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    map_dir = out_dir / "maps"
    for folder in [fig_dir, table_dir, map_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    a_l, _, r_i_l, r_d_l, verts_l, inc_l = calc_arbitrary_pore_properties(L_large, gammas)
    a_s, _, _, _, verts_s, inc_s = calc_arbitrary_pore_properties(L_small, gammas)
    total_volume = a_l * depth_large + a_s * depth_small
    selected_rm = [r_i_l * 1.5, r_i_l * 0.99, r_d_l * 0.99, r_i_l * 0.99 * L_small / L_large, r_d_l * 0.99 * L_small / L_large]
    # Preserve the original CA script's key points exactly.
    _, _, r_i_s, r_d_s, _, _ = calc_arbitrary_pore_properties(L_small, gammas)
    selected_rm = [r_i_l * 1.5, r_i_l * 0.99, r_d_l * 0.99, r_i_s * 0.99, r_d_s * 0.99]

    processes = list(cfg_get("triangle_ca.processes", ["Drainage", "Imbibition"]))
    alpha = float(cfg_get("inversion.alpha", 1.0))
    rows = []
    t2_export = {"T2_Time_ms": t2_axis}
    decay_export = {"Time_ms": times}
    t2_component_export = {"T2_Time_ms": t2_axis}
    decay_component_export = {"Time_ms": times}

    t1_axis = np.logspace(0.0, 3.1, 18)
    t2_det_axis = np.logspace(0.0, 3.1, 24)
    t2_bins_2d = np.logspace(0.5, 3.5, 28)
    tm_mix = float(cfg_get("t2_t2.mixing_time_ms", 30.0))
    b_axis_dt2 = np.concatenate((
        np.asarray(cfg_get("dt2.b_axis", [0.0]), dtype=float),
        np.logspace(float(cfg_get("dt2.b_log_min", -3.0)), float(cfg_get("dt2.b_log_max", 0.55)), int(cfg_get("dt2.b_log_points", 13))),
    ))
    te_axis_dt2 = np.logspace(
        np.log10(float(cfg_get("dt2.te_min_ms", 25.0))),
        np.log10(float(cfg_get("dt2.te_max_ms", 1300.0))),
        int(cfg_get("dt2.te_points", 24)),
    )
    d_bins_dt2 = np.logspace(
        float(cfg_get("dt2.d_log_min", -3.0)),
        math.log10(D_scaled * float(cfg_get("dt2.d_bin_scale_to_bulk_D", 1.5))),
        int(cfg_get("dt2.d_bins", 32)),
    )
    t2_bins_dt2 = np.logspace(
        np.log10(float(cfg_get("dt2.t2_bin_min_ms", 10 ** 0.5))),
        np.log10(float(cfg_get("dt2.t2_bin_max_ms", 10 ** 3.5))),
        int(cfg_get("dt2.t2_bins", 40)),
    )
    dt2_delta = float(cfg_get("dt2.delta_ms", 20.0))
    dt2_pde_dt = float(cfg_get("dt2.pde_dt_ms", 8.0))
    dt2_grad_axis = str(cfg_get("dt2.grad_axis", "y"))
    dt2_alpha = float(cfg_get("dt2.alpha", 0.025))

    for process in processes:
        for idx, rm in enumerate(selected_rm, start=1):
            aw_l = get_single_pore_water_area(L_large, rm, process, theta, gammas)
            aw_s = get_single_pore_water_area(L_small, rm, process, theta, gammas)
            sw = (aw_l * depth_large + aw_s * depth_small) / total_volume
            key = f"P{idx}_{process}_Sw{sw*100:.2f}pct"
            print(f"Running {key}")

            geom_l, l_full = create_pore_geom_arb(a_l, verts_l, inc_l, gammas, a_l - aw_l, theta, False, 0.0)
            geom_s, s_full = create_pore_geom_arb(a_s, verts_s, inc_s, gammas, a_s - aw_s, theta, True, 0.0)
            mesh_l = create_mesh_from_geom(geom_l, a_l)
            mesh_s = create_mesh_from_geom(geom_s, a_s)
            amp_l = solve_single_decay(mesh_l, D_scaled, T2B_scaled, rho_scaled, times, dt) * depth_large
            amp_s = solve_single_decay(mesh_s, D_scaled, T2B_scaled, rho_scaled, times, dt) * depth_small
            amp_unc = amp_l + amp_s
            spec_unc_l = invert_t2_fixed(times, amp_l, t2_axis, alpha)
            spec_unc_s = invert_t2_fixed(times, amp_s, t2_axis, alpha)
            spec_unc = spec_unc_l + spec_unc_s

            geom_l_shift, _ = create_pore_geom_arb(a_l, verts_l, inc_l, gammas, a_l - aw_l, theta, False, L_t / 2.0)
            geom_s_shift, _ = create_pore_geom_arb(a_s, verts_s, inc_s, gammas, a_s - aw_s, theta, True, -L_t / 2.0)
            coupling_active = bool(l_full and s_full)
            if geom_l_shift is not None and geom_s_shift is not None:
                coupled_geom = geom_l_shift + geom_s_shift
                if coupling_active:
                    throat = mt.createRectangle(start=[-W_t / 2.0, -L_t / 2.0], end=[W_t / 2.0, L_t / 2.0])
                    for boundary in throat.boundaries():
                        boundary.setMarker(100)
                    coupled_geom += throat
                coupled_mesh = mt.createMesh(coupled_geom, area=0.05, quality=34)
                amp_c, amp_c_l, amp_c_s = solve_coupled_decay(coupled_mesh, times, dt)
            else:
                coupled_mesh = None
                amp_c = amp_unc.copy()
                amp_c_l = amp_l.copy()
                amp_c_s = amp_s.copy()

            spec_c_l = invert_t2_fixed(times, amp_c_l, t2_axis, alpha) / total_volume
            spec_c_s = invert_t2_fixed(times, amp_c_s, t2_axis, alpha) / total_volume
            spec_c = spec_c_l + spec_c_s

            t2_export[f"{key}_Uncoupled"] = spec_unc / total_volume
            t2_export[f"{key}_Coupled"] = spec_c
            decay_export[f"{key}_Uncoupled"] = amp_unc / total_volume
            decay_export[f"{key}_Coupled"] = amp_c / total_volume
            for mode, large_spec, small_spec, total_spec in [
                ("Uncoupled", spec_unc_l / total_volume, spec_unc_s / total_volume, spec_unc / total_volume),
                ("Coupled", spec_c_l, spec_c_s, spec_c),
            ]:
                t2_component_export[f"{key}_{mode}_Large"] = large_spec
                t2_component_export[f"{key}_{mode}_Small"] = small_spec
                t2_component_export[f"{key}_{mode}_Total"] = total_spec
            for mode, large_decay, small_decay, total_decay in [
                ("Uncoupled", amp_l / total_volume, amp_s / total_volume, amp_unc / total_volume),
                (
                    "Coupled",
                    amp_c_l / total_volume,
                    amp_c_s / total_volume,
                    amp_c / total_volume,
                ),
            ]:
                decay_component_export[f"{key}_{mode}_Large"] = large_decay
                decay_component_export[f"{key}_{mode}_Small"] = small_decay
                decay_component_export[f"{key}_{mode}_Total"] = total_decay
            plot_t2_comparison(fig_dir / f"{safe_name(key)}_T2_uncoupled_vs_coupled.png", key, spec_unc / total_volume, spec_c)

            if coupled_mesh is not None and coupled_mesh.nodeCount() > 5:
                s2 = solve_t2_t2_exchange(coupled_mesh, t1_axis, tm_mix, t2_det_axis)
                s2n = s2 / max(s2[0, 0], 1e-12)
                f2 = invert_t2_t2_nnls(s2n, t1_axis, t2_det_axis, t2_bins_2d, alpha=float(cfg_get("t2_t2.alpha", 0.05)))
                pd.DataFrame(f2.T, index=t2_bins_2d, columns=t2_bins_2d).to_excel(map_dir / f"{safe_name(key)}_Coupled_T2_T2_Map.xlsx")
                plot_map(fig_dir / f"{safe_name(key)}_Coupled_T2_T2.png", f2, t2_bins_2d, t2_bins_2d, f"{key} coupled T2-T2")

                s_dt2_unc = solve_uncoupled_dt2_signal(mesh_l, mesh_s, te_axis_dt2, b_axis_dt2, dt2_pde_dt, dt2_delta, dt2_grad_axis)
                vols_coupled, _, _ = build_coupled_cell_volumes(coupled_mesh)
                s_dt2_c = solve_dt2_signal_pfg(
                    coupled_mesh,
                    vols_coupled,
                    te_axis_dt2,
                    b_axis_dt2,
                    dt2_pde_dt,
                    dt2_delta,
                    dt2_grad_axis,
                    f"Coupled {key}",
                )
                f_dt2_unc = invert_d_t2_nnls(s_dt2_unc, b_axis_dt2, te_axis_dt2, d_bins_dt2, t2_bins_dt2, alpha=dt2_alpha)
                f_dt2_c = invert_d_t2_nnls(s_dt2_c, b_axis_dt2, te_axis_dt2, d_bins_dt2, t2_bins_dt2, alpha=dt2_alpha)
                pd.DataFrame(f_dt2_unc, index=d_bins_dt2, columns=t2_bins_dt2).to_excel(map_dir / f"{safe_name(key)}_Uncoupled_Strict_DT2_Map.xlsx")
                pd.DataFrame(f_dt2_c, index=d_bins_dt2, columns=t2_bins_dt2).to_excel(map_dir / f"{safe_name(key)}_Coupled_Strict_DT2_Map.xlsx")
                pd.DataFrame(s_dt2_unc, index=b_axis_dt2, columns=te_axis_dt2).to_excel(map_dir / f"{safe_name(key)}_Uncoupled_Strict_DT2_Signal.xlsx")
                pd.DataFrame(s_dt2_c, index=b_axis_dt2, columns=te_axis_dt2).to_excel(map_dir / f"{safe_name(key)}_Coupled_Strict_DT2_Signal.xlsx")
                plot_map(fig_dir / f"{safe_name(key)}_Uncoupled_Strict_DT2.png", f_dt2_unc, t2_bins_dt2, d_bins_dt2, f"{key} uncoupled strict D-T2", cmap="viridis")
                plot_map(fig_dir / f"{safe_name(key)}_Coupled_Strict_DT2.png", f_dt2_c, t2_bins_dt2, d_bins_dt2, f"{key} coupled strict D-T2", cmap="viridis")

            rows.append(
                {
                    "key": key,
                    "process": process,
                    "point": idx,
                    "rm_um": rm,
                    "sw_global": sw,
                    "aw_large_um2": aw_l,
                    "aw_small_um2": aw_s,
                    "coupling_active_with_water_throat": coupling_active,
                    "note": "Water throat added only when both pore water domains are full; otherwise coupled geometry is disconnected/weakly connected by actual water geometry only.",
                }
            )

    pd.DataFrame(t2_export).to_excel(table_dir / "AllStates_T2_Uncoupled_Coupled.xlsx", index=False)
    pd.DataFrame(decay_export).to_excel(table_dir / "AllStates_Decay_Uncoupled_Coupled.xlsx", index=False)
    pd.DataFrame(t2_component_export).to_excel(table_dir / "AllStates_T2_Components_Uncoupled_Coupled.xlsx", index=False)
    pd.DataFrame(decay_component_export).to_excel(table_dir / "AllStates_Decay_Components_Uncoupled_Coupled.xlsx", index=False)
    pd.DataFrame(rows).to_excel(table_dir / "AllStates_Summary.xlsx", index=False)
    with (out_dir / "Simulation_Control_Used.json").open("w", encoding="utf-8") as handle:
        json.dump(SETTINGS, handle, indent=2, ensure_ascii=False)
    print(f"Triangle full suite written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
