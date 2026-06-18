"""Run a focused T2 and T2-T2 simulation for coupled ideal triangular pores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygimli as pg
import pygimli.meshtools as mt

from advanced_tools import triangle_contact_angle_full_suite as tri
from simulation_control import SETTINGS


def cfg_get(path: str, default=None):
    value = SETTINGS
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def angle_token(values: list[float]) -> str:
    return "-".join(str(int(v)) if abs(v - int(v)) < 1e-9 else str(v).replace(".", "p") for v in values)


def theta_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def console_path(path: Path, *, cwd: Path | None = None, encoding: str | None = None) -> str:
    base = cwd or Path.cwd()
    try:
        display = path.relative_to(base)
    except ValueError:
        display = path
    text = str(display)
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target_encoding, errors="backslashreplace").decode(target_encoding)


@dataclass
class IdealTriangleT2T2Config:
    output_dir: Path = Path("simulation_outputs/ideal_triangle_t2_t2_CA0deg_angles60-60-60")
    triangle_angles_deg: list[float] = field(
        default_factory=lambda: [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    )
    contact_angle_deg: float = float(cfg_get("triangle_ca.contact_angle_deg", 0.0))
    t2_alpha: float = float(cfg_get("inversion.alpha", 1.0))
    t2_t2_alpha: float = float(cfg_get("t2_t2.alpha", 0.05))
    mixing_time_ms: float = float(cfg_get("t2_t2.mixing_time_ms", 30.0))
    t2_t2_t1_points: int = int(cfg_get("t2_t2.t1_points", 25))
    t2_t2_t2_points: int = int(cfg_get("t2_t2.t2_points", 35))
    t2_t2_bin_points: int = int(cfg_get("t2_t2.bin_points", 35))
    t2_t2_t_axis_min_ms: float = float(cfg_get("t2_t2.t_axis_min_ms", 1.0))
    t2_t2_t_axis_max_ms: float = float(cfg_get("t2_t2.t_axis_max_ms", 10**3.1))
    t2_t2_bin_min_ms: float = float(cfg_get("t2_t2.bin_min_ms", 10**0.5))
    t2_t2_bin_max_ms: float = float(cfg_get("t2_t2.bin_max_ms", 10**3.5))
    mesh_area_um2: float = 0.05
    mesh_quality: float = 34.0


def make_case_config(output_dir: Path | None = None) -> IdealTriangleT2T2Config:
    gammas = [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    theta = float(cfg_get("triangle_ca.contact_angle_deg", 0.0))
    default_output = (
        Path("simulation_outputs")
        / f"ideal_triangle_t2_t2_CA{theta_token(theta)}deg_angles{angle_token(gammas)}"
    )
    return IdealTriangleT2T2Config(output_dir=output_dir or default_output)


def build_uncoupled_triangle_meshes(cfg: IdealTriangleT2T2Config):
    a_l, _, _, _, verts_l, inc_l = tri.calc_arbitrary_pore_properties(tri.L_large, cfg.triangle_angles_deg)
    a_s, _, _, _, verts_s, inc_s = tri.calc_arbitrary_pore_properties(tri.L_small, cfg.triangle_angles_deg)
    geom_l, _ = tri.create_pore_geom_arb(
        a_l, verts_l, inc_l, cfg.triangle_angles_deg, 0.0, cfg.contact_angle_deg, False, 0.0
    )
    geom_s, _ = tri.create_pore_geom_arb(
        a_s, verts_s, inc_s, cfg.triangle_angles_deg, 0.0, cfg.contact_angle_deg, True, 0.0
    )
    return tri.create_mesh_from_geom(geom_l, a_l), tri.create_mesh_from_geom(geom_s, a_s)


def build_coupled_triangle_mesh(cfg: IdealTriangleT2T2Config):
    a_l, _, _, _, verts_l, inc_l = tri.calc_arbitrary_pore_properties(tri.L_large, cfg.triangle_angles_deg)
    a_s, _, _, _, verts_s, inc_s = tri.calc_arbitrary_pore_properties(tri.L_small, cfg.triangle_angles_deg)
    geom_l, _ = tri.create_pore_geom_arb(
        a_l, verts_l, inc_l, cfg.triangle_angles_deg, 0.0, cfg.contact_angle_deg, False, tri.L_t / 2.0
    )
    geom_s, _ = tri.create_pore_geom_arb(
        a_s, verts_s, inc_s, cfg.triangle_angles_deg, 0.0, cfg.contact_angle_deg, True, -tri.L_t / 2.0
    )
    throat = mt.createRectangle(start=[-tri.W_t / 2.0, -tri.L_t / 2.0], end=[tri.W_t / 2.0, tri.L_t / 2.0])
    for boundary in throat.boundaries():
        boundary.setMarker(100)
    return mt.createMesh(geom_l + throat + geom_s, area=cfg.mesh_area_um2, quality=cfg.mesh_quality)


def summarize_mesh(mesh) -> dict[str, int | float]:
    return {
        "node_count": int(mesh.nodeCount()),
        "cell_count": int(mesh.cellCount()),
        "boundary_count": int(mesh.boundaryCount()),
        "total_cell_area_um2": float(np.sum(np.asarray(mesh.cellSizes()))),
    }


def save_mesh_preview(mesh, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    pg.show(mesh, ax=ax, hold=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Coupled ideal equilateral triangular pores")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_decay_plot(path: Path, times_ms: np.ndarray, uncoupled: np.ndarray, coupled: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(times_ms, uncoupled, lw=2.0, color="black", label="Uncoupled")
    ax.plot(times_ms, coupled, lw=2.0, ls="--", color="#c43c39", label="Coupled")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized signal")
    ax.set_title("T2 decay")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_t2_plot(path: Path, t2_axis: np.ndarray, uncoupled: np.ndarray, coupled: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t2_axis, uncoupled, lw=2.0, color="black", label="Uncoupled")
    ax.plot(t2_axis, coupled, lw=2.0, ls="--", color="#c43c39", label="Coupled")
    ax.set_xscale("log")
    ax.set_xlabel("T2 (ms)")
    ax.set_ylabel("Amplitude / total pore volume")
    ax.set_title("Fixed-alpha T2 inversion")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_t2_t2_plot(path: Path, matrix: np.ndarray, bins: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    z = matrix.T
    contour = ax.contourf(bins, bins, z, levels=25, cmap="hot_r")
    ax.plot([bins.min(), bins.max()], [bins.min(), bins.max()], "k--", lw=1.0, alpha=0.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T2 detect (ms)")
    ax.set_ylabel("T2 encode (ms)")
    ax.set_title("Coupled T2-T2 exchange map")
    fig.colorbar(contour, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_run_manifest(
    cfg: IdealTriangleT2T2Config,
    outputs: dict[str, str],
    mesh_summary: dict[str, int | float],
) -> dict[str, object]:
    return {
        "model_dimension": "2D",
        "geometry": {
            "description": "two coupled equilateral triangular pores",
            "triangle_angles_deg": cfg.triangle_angles_deg,
            "contact_angle_deg": cfg.contact_angle_deg,
            "large_base_um": tri.L_large,
            "small_base_um": tri.L_small,
            "large_depth_um": tri.depth_large,
            "small_depth_um": tri.depth_small,
            "throat_length_um": tri.L_t,
            "throat_width_um": tri.W_t,
            "water_state": "full water in both triangular pores and throat",
        },
        "pde_parameters": {
            "diffusion_um2_per_ms": tri.D_scaled,
            "bulk_t2_ms": tri.T2B_scaled,
            "surface_relaxivity_um_per_ms": tri.rho_scaled,
            "time_step_ms": tri.dt,
            "time_max_ms": float(tri.times[-1]),
        },
        "mesh": mesh_summary,
        "inversion": {
            "mode": "fixed",
            "t2_alpha": cfg.t2_alpha,
            "t2_t2_alpha": cfg.t2_t2_alpha,
            "t2_t2_mixing_time_ms": cfg.mixing_time_ms,
        },
        "outputs": outputs,
        "notes": [
            "This is a 2D idealized mesh simulation, not a true 3D pore structure.",
            "Fixed inversion parameters are read from local simulation_control.py defaults unless overridden by CLI output path only.",
        ],
    }


def run_simulation(cfg: IdealTriangleT2T2Config) -> dict[str, object]:
    figure_dir = cfg.output_dir / "figures"
    table_dir = cfg.output_dir / "tables"
    mesh_dir = cfg.output_dir / "mesh"
    for folder in [figure_dir, table_dir, mesh_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    mesh_l, mesh_s = build_uncoupled_triangle_meshes(cfg)
    coupled_mesh = build_coupled_triangle_mesh(cfg)
    mesh_summary = summarize_mesh(coupled_mesh)

    total_volume = (
        tri.calc_arbitrary_pore_properties(tri.L_large, cfg.triangle_angles_deg)[0] * tri.depth_large
        + tri.calc_arbitrary_pore_properties(tri.L_small, cfg.triangle_angles_deg)[0] * tri.depth_small
        + tri.L_t * tri.W_t * ((tri.depth_large + tri.depth_small) / 2.0)
    )

    amp_l = tri.solve_single_decay(mesh_l, tri.D_scaled, tri.T2B_scaled, tri.rho_scaled, tri.times, tri.dt) * tri.depth_large
    amp_s = tri.solve_single_decay(mesh_s, tri.D_scaled, tri.T2B_scaled, tri.rho_scaled, tri.times, tri.dt) * tri.depth_small
    amp_unc = amp_l + amp_s
    amp_c, amp_c_l, amp_c_s = tri.solve_coupled_decay(coupled_mesh, tri.times, tri.dt)

    spec_unc_l = tri.invert_t2_fixed(tri.times, amp_l, tri.t2_axis, cfg.t2_alpha)
    spec_unc_s = tri.invert_t2_fixed(tri.times, amp_s, tri.t2_axis, cfg.t2_alpha)
    spec_unc = (spec_unc_l + spec_unc_s) / total_volume
    spec_c = tri.invert_t2_fixed(tri.times, amp_c, tri.t2_axis, cfg.t2_alpha) / total_volume

    t1_axis = np.logspace(
        math.log10(cfg.t2_t2_t_axis_min_ms), math.log10(cfg.t2_t2_t_axis_max_ms), cfg.t2_t2_t1_points
    )
    t2_det_axis = np.logspace(
        math.log10(cfg.t2_t2_t_axis_min_ms), math.log10(cfg.t2_t2_t_axis_max_ms), cfg.t2_t2_t2_points
    )
    t2_bins = np.logspace(
        math.log10(cfg.t2_t2_bin_min_ms), math.log10(cfg.t2_t2_bin_max_ms), cfg.t2_t2_bin_points
    )
    t2_t2_signal = tri.solve_t2_t2_exchange(coupled_mesh, t1_axis, cfg.mixing_time_ms, t2_det_axis)
    t2_t2_signal_norm = t2_t2_signal / max(t2_t2_signal[0, 0], 1e-12)
    t2_t2_map = tri.invert_t2_t2_nnls(
        t2_t2_signal_norm, t1_axis, t2_det_axis, t2_bins, alpha=cfg.t2_t2_alpha
    )

    mesh_bms = mesh_dir / "ideal_coupled_triangle_mesh.bms"
    coupled_mesh.save(str(mesh_bms))
    mesh_png = figure_dir / "ideal_coupled_triangle_mesh.png"
    decay_png = figure_dir / "ideal_triangle_t2_decay.png"
    t2_png = figure_dir / "ideal_triangle_t2_fixed_alpha.png"
    t2_t2_png = figure_dir / "ideal_triangle_t2_t2_fixed_alpha.png"
    save_mesh_preview(coupled_mesh, mesh_png)
    save_decay_plot(decay_png, tri.times, amp_unc / max(amp_unc[0], 1e-12), amp_c / max(amp_c[0], 1e-12))
    save_t2_plot(t2_png, tri.t2_axis, spec_unc, spec_c)
    save_t2_t2_plot(t2_t2_png, t2_t2_map, t2_bins)

    mesh_summary_csv = table_dir / "ideal_triangle_mesh_summary.csv"
    decay_csv = table_dir / "ideal_triangle_t2_decay.csv"
    t2_csv = table_dir / "ideal_triangle_t2_spectra.csv"
    t2_t2_signal_csv = table_dir / "ideal_triangle_t2_t2_signal.csv"
    t2_t2_map_csv = table_dir / "ideal_triangle_t2_t2_map.csv"
    pd.DataFrame([mesh_summary]).to_csv(mesh_summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "time_ms": tri.times,
            "uncoupled_signal_normalized": amp_unc / max(amp_unc[0], 1e-12),
            "coupled_signal_normalized": amp_c / max(amp_c[0], 1e-12),
        }
    ).to_csv(decay_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "t2_ms": tri.t2_axis,
            "uncoupled_amplitude_per_total_volume": spec_unc,
            "coupled_amplitude_per_total_volume": spec_c,
        }
    ).to_csv(t2_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(t2_t2_signal_norm, index=t1_axis, columns=t2_det_axis).to_csv(t2_t2_signal_csv, encoding="utf-8-sig")
    pd.DataFrame(t2_t2_map.T, index=t2_bins, columns=t2_bins).to_csv(t2_t2_map_csv, encoding="utf-8-sig")

    outputs = {
        "mesh_bms": str(mesh_bms.resolve()),
        "mesh_png": str(mesh_png.resolve()),
        "mesh_summary_csv": str(mesh_summary_csv.resolve()),
        "decay_png": str(decay_png.resolve()),
        "decay_csv": str(decay_csv.resolve()),
        "t2_png": str(t2_png.resolve()),
        "t2_csv": str(t2_csv.resolve()),
        "t2_t2_png": str(t2_t2_png.resolve()),
        "t2_t2_signal_csv": str(t2_t2_signal_csv.resolve()),
        "t2_t2_map_csv": str(t2_t2_map_csv.resolve()),
    }
    manifest_path = cfg.output_dir / "run_manifest.json"
    outputs["manifest_json"] = str(manifest_path.resolve())
    manifest = build_run_manifest(cfg, outputs, mesh_summary)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused ideal coupled-triangle T2 and T2-T2 simulation.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_case_config(args.output_dir)
    manifest = run_simulation(cfg)
    print(f"Ideal triangle T2/T2-T2 outputs written to: {console_path(cfg.output_dir)}")
    print(f"Manifest: {console_path(Path(str(manifest['outputs']['manifest_json'])))}")


if __name__ == "__main__":
    main()
