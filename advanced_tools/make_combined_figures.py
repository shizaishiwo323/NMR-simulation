"""Create combined comparison figures from completed simulation outputs."""

from __future__ import annotations

import re
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from simulation_control import SETTINGS
from triangle_contact_angle_full_suite import (
    L_large,
    L_small,
    calc_arbitrary_pore_properties,
    cfg_get,
    create_mesh_from_geom,
    create_pore_geom_arb,
    depth_large,
    depth_small,
    get_single_pore_water_area,
)

try:
    import pygimli as pg
except Exception:
    pg = None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if not value:
        return default
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _angle_token(values: list[float]) -> str:
    return "-".join(str(int(v)) if abs(v - int(v)) < 1e-9 else str(v).replace(".", "p") for v in values)


def _theta_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


_env_theta = _env_float("NMR_CONTACT_ANGLE_DEG", float(SETTINGS["triangle_ca"].get("contact_angle_deg", 0.0)))
_env_gammas = _env_float_list(
    "NMR_TRIANGLE_ANGLES_DEG",
    [float(v) for v in SETTINGS["triangle_ca"].get("triangle_angles_deg", [60.0, 60.0, 60.0])],
)
SETTINGS["triangle_ca"]["contact_angle_deg"] = _env_theta
SETTINGS["triangle_ca"]["triangle_angles_deg"] = _env_gammas
SETTINGS["geometry"]["contact_angle_deg"] = _env_theta
SETTINGS["geometry"]["triangle_angles_deg"] = _env_gammas

_default_case_suffix = f"CA{_theta_token(_env_theta)}deg_angles_{_angle_token(_env_gammas)}"

TRI_ROOT = Path(os.environ.get("NMR_TRI_ROOT", f"simulation_outputs/triangle_full_{_default_case_suffix}"))
TRI_TABLES = TRI_ROOT / "tables"
TRI_MAPS = TRI_ROOT / "maps"
IMG_ROOT = Path("simulation_outputs/image_slice_Result")
CA_BASE = Path(os.environ.get("NMR_CA_BASE", f"simulation_outputs/triangle_{_angle_token(_env_gammas).replace('-', '_')}_contact_{int(_env_theta)}"))
OUT = Path(os.environ.get("NMR_COMBINED_OUT", f"simulation_outputs/combined_figures_{_default_case_suffix}"))


def parse_key(key: str) -> tuple[str, int, float]:
    match = re.match(r"P(\d+)_(Drainage|Imbibition)_Sw([0-9.]+)pct", key)
    if not match:
        raise ValueError(f"Cannot parse key: {key}")
    return match.group(2), int(match.group(1)), float(match.group(3))


def normalize(values: np.ndarray) -> np.ndarray:
    vmax = float(np.nanmax(values)) if values.size else 0.0
    return values / vmax if vmax > 1e-12 else values


def read_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_excel(path, index_col=0)
    y_axis = frame.index.to_numpy(dtype=float)
    x_axis = frame.columns.to_numpy(dtype=float)
    values = frame.to_numpy(dtype=float)
    return x_axis, y_axis, values


def setup_grid_title(fig, title: str) -> None:
    fig.suptitle(title, fontsize=16, fontweight="bold")


def table_path(name: str) -> Path:
    source = TRI_TABLES / name
    corrected = source.with_name(source.stem + "_Corrected.xlsx")
    return corrected if corrected.exists() else source


def plot_t2_grid(summary: pd.DataFrame, t2_frame: pd.DataFrame) -> Path:
    t2 = t2_frame["T2_Time_ms"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.8), sharex=True, sharey=True)
    setup_grid_title(fig, "All states: uncoupled vs coupled T2 spectra")
    for _, row in summary.iterrows():
        process, point, sw = parse_key(row["key"])
        ax = axes[0 if process == "Drainage" else 1, point - 1]
        unc = t2_frame[f"{row['key']}_Uncoupled"].to_numpy(dtype=float)
        coup = t2_frame[f"{row['key']}_Coupled"].to_numpy(dtype=float)
        ax.plot(t2, unc, color="0.25", lw=1.8, label="Uncoupled")
        ax.plot(t2, coup, color="#c9252d", lw=1.8, ls="--", label="Coupled")
        ax.set_xscale("log")
        ax.set_title(f"{process[:3]} P{point}\nSw={sw:.1f}%")
        ax.grid(alpha=0.25, which="both")
        if point == 1:
            ax.set_ylabel("T2 amplitude / total pore volume")
        if process == "Imbibition":
            ax.set_xlabel("T2 (ms)")
        if process == "Drainage" and point == 1:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_01_AllStates_T2_Uncoupled_vs_Coupled.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_component_curves(ax, x, frame, key: str, mode: str):
    large = frame[f"{key}_{mode}_Large"].to_numpy(dtype=float)
    small = frame[f"{key}_{mode}_Small"].to_numpy(dtype=float)
    total = frame[f"{key}_{mode}_Total"].to_numpy(dtype=float)
    linestyle = "-" if mode == "Uncoupled" else "--"
    alpha = 0.95 if mode == "Uncoupled" else 0.85
    ax.plot(x, large, color="#2ca25f", lw=1.5, ls=linestyle, alpha=alpha, label=f"{mode} Large")
    ax.plot(x, small, color="#fdae6b", lw=1.5, ls=linestyle, alpha=alpha, label=f"{mode} Small")
    ax.plot(x, total, color="k" if mode == "Uncoupled" else "#c9252d", lw=1.8, ls=linestyle, alpha=alpha, label=f"{mode} Total")


def plot_large_small_only(ax, x, frame, key: str):
    unc_l = frame[f"{key}_Uncoupled_Large"].to_numpy(dtype=float)
    unc_s = frame[f"{key}_Uncoupled_Small"].to_numpy(dtype=float)
    coup_l = frame[f"{key}_Coupled_Large"].to_numpy(dtype=float)
    coup_s = frame[f"{key}_Coupled_Small"].to_numpy(dtype=float)
    ax.fill_between(x, unc_l, color="#2ca25f", alpha=0.35, label="Uncoupled Large")
    ax.fill_between(x, unc_s, color="#fdae6b", alpha=0.45, label="Uncoupled Small")
    ax.plot(x, coup_l, color="#2ca25f", lw=1.8, ls="--", label="Coupled Large")
    ax.plot(x, coup_s, color="#fdae6b", lw=1.8, ls="--", label="Coupled Small")
    ymax = float(np.nanmax([np.nanmax(unc_l), np.nanmax(unc_s), np.nanmax(coup_l), np.nanmax(coup_s)]))
    ax.set_ylim(0, ymax * 1.2 if ymax > 1e-12 else 1e-4)


def plot_water_distribution_with_t2(summary: pd.DataFrame, t2_component_frame: pd.DataFrame) -> Path:
    gammas = [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    theta = float(cfg_get("triangle_ca.contact_angle_deg", 0.0))
    area_l, _, r_i_l, r_d_l, verts_l, inc_l = calc_arbitrary_pore_properties(L_large, gammas)
    area_s, _, r_i_s, r_d_s, verts_s, inc_s = calc_arbitrary_pore_properties(L_small, gammas)
    total_volume = area_l * depth_large + area_s * depth_small
    selected_rm = [r_i_l * 1.5, r_i_l * 0.99, r_d_l * 0.99, r_i_s * 0.99, r_d_s * 0.99]
    t2 = t2_component_frame["T2_Time_ms"].to_numpy(dtype=float)

    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle("Water distribution and corresponding T2 spectra", fontsize=16, fontweight="bold")
    row_for = {("Drainage", "morph"): 0, ("Drainage", "t2"): 1, ("Imbibition", "morph"): 2, ("Imbibition", "t2"): 3}

    for process in ["Drainage", "Imbibition"]:
        for idx, rm in enumerate(selected_rm, start=1):
            aw_l = get_single_pore_water_area(L_large, rm, process, theta, gammas)
            aw_s = get_single_pore_water_area(L_small, rm, process, theta, gammas)
            sw = (aw_l * depth_large + aw_s * depth_small) / total_volume
            key = f"P{idx}_{process}_Sw{sw*100:.2f}pct"

            ax_m = axes[row_for[(process, "morph")], idx - 1]
            geom_l, _ = create_pore_geom_arb(area_l, verts_l, inc_l, gammas, area_l - aw_l, theta, False, 0.0)
            geom_s, _ = create_pore_geom_arb(area_s, verts_s, inc_s, gammas, area_s - aw_s, theta, True, 0.0)
            mesh_l = create_mesh_from_geom(geom_l, area_l)
            mesh_s = create_mesh_from_geom(geom_s, area_s)

            v_l = verts_l + [verts_l[0]]
            v_s = [[v[0], -v[1]] for v in verts_s] + [[verts_s[0][0], -verts_s[0][1]]]
            ax_m.plot([v[0] for v in v_l], [v[1] for v in v_l], "k-", lw=1.0)
            ax_m.plot([v[0] for v in v_s], [v[1] for v in v_s], "k-", lw=1.0)
            if pg is not None:
                if mesh_l.nodeCount() > 5:
                    pg.show(mesh_l, ax=ax_m, hold=True)
                if mesh_s.nodeCount() > 5:
                    pg.show(mesh_s, ax=ax_m, hold=True)
            ax_m.set_aspect("equal")
            ax_m.set_xlim(-13, 13)
            ax_m.set_ylim(-9, 20)
            ax_m.set_title(f"{process[:3]} P{idx}: Sw={sw*100:.1f}%", fontsize=10)
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            if idx == 1:
                ax_m.set_ylabel(f"{process}\nwater distribution")

            ax_t = axes[row_for[(process, "t2")], idx - 1]
            plot_large_small_only(ax_t, t2, t2_component_frame, key)
            ax_t.set_xscale("log")
            ax_t.set_xlim(1, 5000)
            ax_t.grid(alpha=0.25, which="both")
            if idx == 1:
                ax_t.set_ylabel("large/small T2\namp. / total pore vol.")
            if process == "Imbibition":
                ax_t.set_xlabel("T2 (ms)")
            if process == "Drainage" and idx == 1:
                ax_t.legend(fontsize=6, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "Combined_00_WaterDistribution_and_T2.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_component_t2_grid(summary: pd.DataFrame, t2_component_frame: pd.DataFrame) -> Path:
    t2 = t2_component_frame["T2_Time_ms"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.8), sharex=True)
    setup_grid_title(fig, "Component T2 spectra: Large / Small / Total, uncoupled vs coupled")
    for _, row in summary.iterrows():
        process, point, sw = parse_key(row["key"])
        ax = axes[0 if process == "Drainage" else 1, point - 1]
        plot_component_curves(ax, t2, t2_component_frame, row["key"], "Uncoupled")
        plot_component_curves(ax, t2, t2_component_frame, row["key"], "Coupled")
        ax.set_xscale("log")
        ax.set_xlim(1, 5000)
        ax.set_title(f"{process[:3]} P{point}\nSw={sw:.1f}%")
        ax.grid(alpha=0.25, which="both")
        if point == 1:
            ax.set_ylabel("T2 amplitude / total pore volume")
        if process == "Imbibition":
            ax.set_xlabel("T2 (ms)")
        if process == "Drainage" and point == 1:
            ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_08_Component_T2_LargeSmallTotal.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_component_decay_grid(summary: pd.DataFrame, decay_component_frame: pd.DataFrame) -> Path:
    time = decay_component_frame["Time_ms"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.8), sharex=True)
    setup_grid_title(fig, "Component decay curves: Large / Small / Total, uncoupled vs coupled")
    for _, row in summary.iterrows():
        process, point, sw = parse_key(row["key"])
        ax = axes[0 if process == "Drainage" else 1, point - 1]
        plot_component_curves(ax, time, decay_component_frame, row["key"], "Uncoupled")
        plot_component_curves(ax, time, decay_component_frame, row["key"], "Coupled")
        ax.set_title(f"{process[:3]} P{point}\nSw={sw:.1f}%")
        ax.grid(alpha=0.25)
        if point == 1:
            ax.set_ylabel("signal / total pore volume")
        if process == "Imbibition":
            ax.set_xlabel("time (ms)")
        if process == "Drainage" and point == 1:
            ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_09_Component_Decay_LargeSmallTotal.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decay_grid(summary: pd.DataFrame, decay_frame: pd.DataFrame) -> Path:
    time = decay_frame["Time_ms"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.8), sharex=True, sharey=True)
    setup_grid_title(fig, "All states: uncoupled vs coupled decay")
    for _, row in summary.iterrows():
        process, point, sw = parse_key(row["key"])
        ax = axes[0 if process == "Drainage" else 1, point - 1]
        unc = decay_frame[f"{row['key']}_Uncoupled"].to_numpy(dtype=float)
        coup = decay_frame[f"{row['key']}_Coupled"].to_numpy(dtype=float)
        ax.plot(time, unc, color="0.25", lw=1.8, label="Uncoupled")
        ax.plot(time, coup, color="#c9252d", lw=1.8, ls="--", label="Coupled")
        ax.set_title(f"{process[:3]} P{point}\nSw={sw:.1f}%")
        ax.grid(alpha=0.25)
        if point == 1:
            ax.set_ylabel("signal / total pore volume")
        if process == "Imbibition":
            ax.set_xlabel("time (ms)")
        if process == "Drainage" and point == 1:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_02_AllStates_Decay_Uncoupled_vs_Coupled.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_summary(summary: pd.DataFrame, t2_component_frame: pd.DataFrame) -> Path:
    t2 = t2_component_frame["T2_Time_ms"].to_numpy(dtype=float)
    rows = []
    for _, row in summary.iterrows():
        process, point, sw = parse_key(row["key"])
        for mode in ["Uncoupled", "Coupled"]:
            for component in ["Large", "Small"]:
                spec = t2_component_frame[f"{row['key']}_{mode}_{component}"].to_numpy(dtype=float)
                area = float(np.trapz(spec, np.log10(t2)))
                peak_t2 = float(t2[int(np.nanargmax(spec))]) if np.nanmax(spec) > 0 else np.nan
                rows.append(
                    {
                        "process": process,
                        "point": point,
                        "sw": sw,
                        "mode": mode,
                        "component": component,
                        "area": area,
                        "peak_t2": peak_t2,
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_excel(OUT / "Combined_T2_Component_Peak_Area_Metrics.xlsx", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    style = {
        ("Drainage", "Uncoupled"): ("#1f77b4", "o", "-"),
        ("Drainage", "Coupled"): ("#1f77b4", "o", "--"),
        ("Imbibition", "Uncoupled"): ("#d62728", "s", "-"),
        ("Imbibition", "Coupled"): ("#d62728", "s", "--"),
    }
    for col, component in enumerate(["Large", "Small"]):
        ax_peak = axes[0, col]
        ax_area = axes[1, col]
        for process in ["Drainage", "Imbibition"]:
            for mode in ["Uncoupled", "Coupled"]:
                color, marker, ls = style[(process, mode)]
                sub = metrics[
                    (metrics["process"] == process)
                    & (metrics["mode"] == mode)
                    & (metrics["component"] == component)
                ].sort_values("point")
                label = f"{process} {mode}"
                ax_peak.plot(sub["sw"], sub["peak_t2"], marker=marker, color=color, ls=ls, lw=1.8, label=label)
                ax_area.plot(sub["sw"], sub["area"], marker=marker, color=color, ls=ls, lw=1.8, label=label)
        ax_peak.set_yscale("log")
        ax_peak.set_title(f"{component} pore peak T2")
        ax_peak.set_ylabel("peak T2 (ms)")
        ax_peak.grid(alpha=0.3, which="both")
        ax_area.set_title(f"{component} pore spectral area")
        ax_area.set_xlabel("Sw (%)")
        ax_area.set_ylabel("log-T2 spectral area")
        ax_area.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Component metrics: Large/Small peak T2 and spectral area", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "Combined_03_T2_Metric_Summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_t2t2_grid(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    setup_grid_title(fig, "Coupled T2-T2 maps for all drainage/imbibition states")
    for _, row in summary.iterrows():
        process, point, sw = parse_key(row["key"])
        ax = axes[0 if process == "Drainage" else 1, point - 1]
        path = TRI_MAPS / f"{row['key']}_Coupled_T2_T2_Map.xlsx"
        x, y, z = read_map(path)
        ax.contourf(x, y, normalize(z).T, levels=20, cmap="hot_r")
        ax.plot([x[0], x[-1]], [x[0], x[-1]], "k--", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{process[:3]} P{point}\nSw={sw:.1f}%")
        if point == 1:
            ax.set_ylabel("T2 encode (ms)")
        if process == "Imbibition":
            ax.set_xlabel("T2 detect (ms)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_04_AllStates_Coupled_T2T2_Maps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dt2_selected(summary: pd.DataFrame) -> Path:
    selected = []
    for process, points in [("Drainage", [1, 3, 5]), ("Imbibition", [1, 2, 5])]:
        for point in points:
            match = None
            for key in summary["key"].astype(str):
                key_process, key_point, _ = parse_key(key)
                if key_process == process and key_point == point:
                    match = key
                    break
            if match is not None:
                selected.append(match)
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 18))
    setup_grid_title(fig, "Strict D-T2 maps: uncoupled vs coupled selected states")
    for r, key in enumerate(selected):
        process, point, sw = parse_key(key)
        for c, mode in enumerate(["Uncoupled", "Coupled"]):
            ax = axes[r, c]
            path = TRI_MAPS / f"{key}_{mode}_Strict_DT2_Map.xlsx"
            x, y, z = read_map(path)
            ax.contourf(x, y, normalize(z), levels=20, cmap="viridis")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{key}\n{mode}")
            ax.set_xlabel("T2 (ms)")
            ax.set_ylabel("D app (um2/ms)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "Combined_05_Selected_Strict_DT2_Uncoupled_vs_Coupled.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_glass_bead_panel() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("Glass-bead segmented image: smoothing, automatic regularization, and maps", fontsize=15, fontweight="bold")
    preview = Image.open(IMG_ROOT / "Result_cleaned_preview.png")
    axes[0, 0].imshow(preview)
    axes[0, 0].set_title("cleaned three-phase image")
    axes[0, 0].axis("off")

    lcurve = pd.read_excel(IMG_ROOT / "ImageSlice_LCurve_vs_FixedAlpha.xlsx")
    axes[0, 1].plot(lcurve["t2_ms"], lcurve["coupled_lcurve"], "k-", lw=2, label="L-curve")
    axes[0, 1].plot(lcurve["t2_ms"], lcurve["coupled_fixed"], "r--", lw=2, label="fixed")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("T2: L-curve vs fixed")
    axes[0, 1].set_xlabel("T2 (ms)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, which="both")

    decay = pd.read_excel(IMG_ROOT / "ImageSlice_T2_Decay.xlsx")
    axes[0, 2].plot(decay["time_ms"], decay["uncoupled_total"], color="0.25", label="uncoupled")
    axes[0, 2].plot(decay["time_ms"], decay["coupled_total"], color="#c9252d", ls="--", label="coupled")
    axes[0, 2].set_title("T2 decay")
    axes[0, 2].set_xlabel("time (ms)")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    for ax, file_name, title, cmap in [
        (axes[1, 0], "ImageSlice_Coupled_T2_T2_Map.xlsx", "T2-T2 map", "hot_r"),
        (axes[1, 1], "ImageSlice_Coupled_DT2_Map.xlsx", "D-T2 map", "viridis"),
        (axes[1, 2], "ImageSlice_Uncoupled_DT2_Map.xlsx", "uncoupled D-T2", "viridis"),
    ]:
        x, y, z = read_map(IMG_ROOT / file_name)
        ax.contourf(x, y, normalize(z), levels=20, cmap=cmap)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("T2 (ms)")
        ax.set_ylabel("D/T2 axis")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_06_GlassBead_ImageSlice_Panel.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_article_overview(summary: pd.DataFrame, t2_frame: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle("Proposed article overview: hysteresis, coupling, and complex pores", fontsize=16, fontweight="bold")

    ax0 = fig.add_subplot(gs[0, 0])
    wrc = Image.open(CA_BASE / "Figure_1_Water_Retention_Curve.png")
    ax0.imshow(wrc)
    ax0.set_title("A. Drainage/imbibition path")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    t2 = t2_frame["T2_Time_ms"].to_numpy(dtype=float)
    for key, color in [("P3_Drainage_Sw17.54pct", "#1f77b4"), ("P3_Imbibition_Sw17.54pct", "#d62728")]:
        ax1.plot(t2, t2_frame[f"{key}_Uncoupled"].to_numpy(dtype=float), color=color, lw=2, label=key.replace("_", " "))
    ax1.set_xscale("log")
    ax1.set_title("B. Same Sw, path-dependent T2")
    ax1.set_xlabel("T2 (ms)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")

    ax2 = fig.add_subplot(gs[0, 2])
    key = "P1_Drainage_Sw100.00pct"
    ax2.plot(t2, t2_frame[f"{key}_Uncoupled"].to_numpy(dtype=float), "0.25", lw=2, label="uncoupled")
    ax2.plot(t2, t2_frame[f"{key}_Coupled"].to_numpy(dtype=float), "#c9252d", ls="--", lw=2, label="coupled")
    ax2.set_xscale("log")
    ax2.set_title("C. Two-pore coupling")
    ax2.set_xlabel("T2 (ms)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    for ax, file_name, title, cmap in [
        (fig.add_subplot(gs[1, 0]), TRI_MAPS / f"{key}_Coupled_T2_T2_Map.xlsx", "D. T2-T2 exchange", "hot_r"),
        (fig.add_subplot(gs[1, 1]), TRI_MAPS / f"{key}_Coupled_Strict_DT2_Map.xlsx", "E. strict D-T2", "viridis"),
    ]:
        x, y, z = read_map(file_name)
        ax.contourf(x, y, normalize(z), levels=20, cmap=cmap)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("T2 (ms)")
        ax.set_ylabel("D/T2 axis")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(Image.open(IMG_ROOT / "Result_cleaned_preview.png"))
    ax5.set_title("F. real glass-bead geometry")
    ax5.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "Combined_07_Article_Overview_Draft.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_excel(TRI_TABLES / "AllStates_Summary.xlsx")
    t2_frame = pd.read_excel(table_path("AllStates_T2_Uncoupled_Coupled.xlsx"))
    decay_frame = pd.read_excel(table_path("AllStates_Decay_Uncoupled_Coupled.xlsx"))
    t2_component_frame = pd.read_excel(table_path("AllStates_T2_Components_Uncoupled_Coupled.xlsx"))
    decay_component_frame = pd.read_excel(table_path("AllStates_Decay_Components_Uncoupled_Coupled.xlsx"))
    paths = [
        plot_water_distribution_with_t2(summary, t2_component_frame),
        plot_t2_grid(summary, t2_frame),
        plot_decay_grid(summary, decay_frame),
        plot_metric_summary(summary, t2_component_frame),
        plot_t2t2_grid(summary),
        plot_dt2_selected(summary),
        plot_component_t2_grid(summary, t2_component_frame),
        plot_component_decay_grid(summary, decay_component_frame),
    ]
    glass_required = [
        IMG_ROOT / "Result_cleaned_preview.png",
        IMG_ROOT / "ImageSlice_LCurve_vs_FixedAlpha.xlsx",
        IMG_ROOT / "ImageSlice_T2_Decay.xlsx",
        IMG_ROOT / "ImageSlice_Coupled_T2_T2_Map.xlsx",
        IMG_ROOT / "ImageSlice_Coupled_DT2_Map.xlsx",
        IMG_ROOT / "ImageSlice_Uncoupled_DT2_Map.xlsx",
    ]
    if all(path.exists() for path in glass_required):
        paths.append(plot_glass_bead_panel())
    article_required = [
        CA_BASE / "Figure_1_Water_Retention_Curve.png",
        IMG_ROOT / "Result_cleaned_preview.png",
    ]
    if all(path.exists() for path in article_required):
        paths.append(plot_article_overview(summary, t2_frame))
    pd.DataFrame({"figure": [str(path) for path in paths]}).to_excel(OUT / "Combined_Figure_Index.xlsx", index=False)
    print("Combined figures written:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
