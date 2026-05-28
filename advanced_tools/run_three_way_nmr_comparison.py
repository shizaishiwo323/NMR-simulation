r"""Batch three-way NMR comparison workflow for RTSPHEM experiment folders.

Usage examples, from the ``conda ml`` environment:

    python C:\Users\imgw\Documents\Codex\NMR模拟\advanced_tools\run_three_way_nmr_comparison.py ^
        "C:\Users\imgw\Documents\Codex\RTSPHEM-main\outputs\rtm_batches\33\exp_015"

Or edit DEFAULT_EXP_DIR below and run without command-line arguments:

    python C:\Users\imgw\Documents\Codex\NMR模拟\advanced_tools\run_three_way_nmr_comparison.py

The script expects an experiment directory with the usual RTSPHEM layout:

    exp_xxx/
      run_metadata.json
      global_evolution_log.csv
      interface_images/timestep_0001.png ...
      comsol_results/T2_t0001.xlsx ...
      individual_plots/concentration/concentration_0001.png ...

It creates a timestamped output directory inside the experiment directory and
does not delete or overwrite prior runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# 用户可改参数入口
# =============================================================================

# 0. 默认输入路径
# 以后换实验时，通常只需要把这里改成新的 exp_xxx 目录即可。
# 命令行如果额外传入路径，会覆盖这个默认路径。
DEFAULT_EXP_DIR = Path(r"C:\Users\imgw\Documents\Codex\RTSPHEM-main\outputs\rtm_batches\33\exp_015")

# 1. 物理参数
# COMSOL 模型里使用 D=2e-9 m^2/s，即 2 um^2/ms。
DIFFUSION_UM2_PER_MS = 2.0

# 体弛豫时间。COMSOL 里 T2=3 s，因此这里写 3000 ms。
BULK_T2_MS = 3000.0

# 水-固界面的 surface relaxivity。COMSOL rho=5e-5 m/s，
# 换算为 0.05 um/ms。这个参数对 T2 峰位置影响很大。
RHO_SOLID_UM_PER_MS = 0.05

# 水-气界面弛豫率。当前对比里设为 0，表示气液边界不额外贡献表面弛豫。
RHO_GAS_UM_PER_MS = 0.0


# 2. 时间离散参数
# T2 衰减求解的时间步长和最大时间。越小/越长越慢。
DECAY_DT_MS = 5.0
DECAY_T_MAX_MS = 5500.0


# 3. PNG 图像到物理尺寸的换算
# True 表示：metadata 里的 lengthXAxis_cm / lengthYAxis_cm 对应 PNG 中
# 非白色求解域的 bbox，而不是整张 PNG 的宽高。这是前面 COMSOL/PNG
# 尺寸差异排查后的结论。
USE_NONWHITE_BBOX_FOR_GEOMETRY_SIZE = True

# 求解前把最大图像维度缩到这个值以内。None 表示不缩放。
# 500 是前面批量计算中使用过的折中值。
MAX_GRID_SIZE = 500


# 4. 三角网格参数
# mesh_bulk_size 控制水域内部点间距，mesh_boundary_size 控制边界点间距。
# 数值越小，网格越细、越慢。前面 exp_013 批量验证使用 16/5。
TRI_MESH_BULK_SIZE_UM = 16.0
TRI_MESH_BOUNDARY_SIZE_UM = 5.0

# 防止三角网格节点数过大导致运行时间失控。
TRI_MESH_MAX_POINTS = 25000

# conda ml 环境若没有 pyGIMLi，脚本会自动用这个 Python 跑三角网格部分。
# 这台机器上 C:\Python314\python.exe 已验证安装了 pyGIMLi。
PYGIMLI_FALLBACK_PYTHON = Path(r"C:\Python314\python.exe")


# 5. T2 反演参数
# 按用户要求固定 alpha=0.1，T2 范围 1-100000 ms。
NNLS_ALPHA = 0.1
T2_MIN_MS = 1.0
T2_MAX_MS = 100000.0
T2_NUM_BINS = 240


# 6. 视频参数
VIDEO_FPS = 8


# =============================================================================
# 项目内模块导入
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from advanced_tools.compare_png_nmr_with_comsol import (  # noqa: E402
    invert_fixed_alpha,
    normalize,
    read_comsol_probe_excel,
    read_png_decay_csv,
    trim_to_decay,
)
from advanced_tools.png_phase_nmr_decay import (  # noqa: E402
    OUTSIDE,
    SimulationParams,
    classify_png,
    simulate_png,
)

try:
    import pygimli as pg  # noqa: F401

    CURRENT_PYTHON_HAS_PYGIMLI = True
except Exception:
    CURRENT_PYTHON_HAS_PYGIMLI = False


def info(message: str) -> None:
    print(f"[three-way-nmr] {message}", flush=True)


def alpha_token(value: float) -> str:
    return f"alpha{value:g}".replace(".", "p").replace("-", "m")


def timestep_from_name(path: Path) -> int | None:
    match = re.search(r"(?:timestep_|T2_t|concentration_)(\d+)", path.stem)
    return int(match.group(1)) if match else None


def read_metadata(exp_dir: Path) -> dict:
    metadata_path = exp_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def compute_pixel_size_from_first_png(exp_dir: Path, metadata: dict) -> tuple[float, float, dict]:
    """Return raw-pixel physical sizes in um.

    If USE_NONWHITE_BBOX_FOR_GEOMETRY_SIZE is True, the physical length in
    metadata is applied to the non-white sample bbox. This matches the COMSOL
    geometry definition used in the previous manual comparison.
    """

    first_png = next((exp_dir / "interface_images").glob("timestep_*.png"))
    rgb = np.asarray(Image.open(first_png).convert("RGB"))
    labels = classify_png(rgb)
    height, width = labels.shape

    length_x_um = float(metadata["parameters"]["lengthXAxis_cm"]) * 10000.0
    length_y_um = float(metadata["parameters"]["lengthYAxis_cm"]) * 10000.0

    if USE_NONWHITE_BBOX_FOR_GEOMETRY_SIZE:
        yy, xx = np.nonzero(labels != OUTSIDE)
        if yy.size == 0 or xx.size == 0:
            raise ValueError(f"No non-white geometry bbox found in {first_png}")
        bbox = {
            "row_min": int(yy.min()),
            "row_max": int(yy.max()),
            "col_min": int(xx.min()),
            "col_max": int(xx.max()),
            "bbox_height_px": int(yy.max() - yy.min() + 1),
            "bbox_width_px": int(xx.max() - xx.min() + 1),
            "raw_height_px": int(height),
            "raw_width_px": int(width),
            "geometry_size_mode": "nonwhite_bbox",
        }
        pixel_size_x_um = length_x_um / bbox["bbox_width_px"]
        pixel_size_y_um = length_y_um / bbox["bbox_height_px"]
    else:
        bbox = {
            "raw_height_px": int(height),
            "raw_width_px": int(width),
            "geometry_size_mode": "full_png",
        }
        pixel_size_x_um = length_x_um / width
        pixel_size_y_um = length_y_um / height

    bbox["length_x_um"] = length_x_um
    bbox["length_y_um"] = length_y_um
    bbox["pixel_size_x_um"] = pixel_size_x_um
    bbox["pixel_size_y_um"] = pixel_size_y_um
    return pixel_size_x_um, pixel_size_y_um, bbox


def build_params(pixel_size_x_um: float, pixel_size_y_um: float, solver: str) -> SimulationParams:
    return SimulationParams(
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        diffusion_um2_per_ms=DIFFUSION_UM2_PER_MS,
        bulk_t2_ms=BULK_T2_MS,
        rho_solid_um_per_ms=RHO_SOLID_UM_PER_MS,
        rho_gas_um_per_ms=RHO_GAS_UM_PER_MS,
        dt_ms=DECAY_DT_MS,
        t_max_ms=DECAY_T_MAX_MS,
        max_grid_size=MAX_GRID_SIZE,
        solver=solver,
        mesh_bulk_size_um=TRI_MESH_BULK_SIZE_UM,
        mesh_boundary_size_um=TRI_MESH_BOUNDARY_SIZE_UM,
        mesh_max_points=TRI_MESH_MAX_POINTS,
    )


def collect_aligned_steps(exp_dir: Path, max_steps: int | None) -> list[dict]:
    comsol_dir = exp_dir / "comsol_results"
    png_dir = exp_dir / "interface_images"
    concentration_dir = exp_dir / "individual_plots" / "concentration"

    comsol = {timestep_from_name(p): p for p in comsol_dir.glob("T2_t*.xlsx")}
    pngs = {timestep_from_name(p): p for p in png_dir.glob("timestep_*.png")}
    conc = {timestep_from_name(p): p for p in concentration_dir.glob("concentration_*.png")}
    common_steps = sorted(set(comsol) & set(pngs) & set(conc))
    common_steps = [step for step in common_steps if step is not None]
    if max_steps is not None:
        common_steps = common_steps[:max_steps]
    if not common_steps:
        raise FileNotFoundError("No aligned COMSOL, PNG, and concentration timesteps found.")
    return [
        {
            "timestep": step,
            "comsol_xlsx": comsol[step],
            "png": pngs[step],
            "concentration_png": conc[step],
        }
        for step in common_steps
    ]


def read_time_metadata(exp_dir: Path) -> dict[int, dict]:
    path = exp_dir / "global_evolution_log.csv"
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(float(row["timestep"]))
            rows[step] = row
    return rows


def run_pixel_decay(steps: list[dict], output_dir: Path, params: SimulationParams) -> dict[int, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[int, float] = {}
    for item in steps:
        step = item["timestep"]
        info(f"pixel decay timestep {step:04d}")
        start = time.perf_counter()
        simulate_png(item["png"], output_dir, params)
        timings[step] = time.perf_counter() - start
    return timings


def run_triangular_decay_direct(steps: list[dict], output_dir: Path, params: SimulationParams) -> dict[int, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[int, float] = {}
    for item in steps:
        step = item["timestep"]
        info(f"triangular decay timestep {step:04d}")
        start = time.perf_counter()
        simulate_png(item["png"], output_dir, params)
        timings[step] = time.perf_counter() - start
    return timings


def run_triangular_decay_subprocess(
    steps: list[dict],
    output_dir: Path,
    params: SimulationParams,
    python_exe: Path,
) -> dict[int, float]:
    """Run triangular pyGIMLi solve through a Python that has pyGIMLi.

    This keeps the wrapper runnable from conda ml even when ml itself does not
    have pyGIMLi installed.
    """

    if not python_exe.exists():
        raise FileNotFoundError(
            "Current Python cannot import pyGIMLi and fallback Python is missing: "
            f"{python_exe}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    script = PROJECT_DIR / "advanced_tools" / "png_phase_nmr_decay.py"
    timings: dict[int, float] = {}
    for item in steps:
        step = item["timestep"]
        info(f"triangular decay timestep {step:04d} via {python_exe}")
        command = [
            str(python_exe),
            str(script),
            str(item["png"]),
            "--output-dir",
            str(output_dir),
            "--pixel-size-x-um",
            str(params.pixel_size_x_um),
            "--pixel-size-y-um",
            str(params.pixel_size_y_um),
            "--diffusion-um2-per-ms",
            str(params.diffusion_um2_per_ms),
            "--bulk-t2-ms",
            str(params.bulk_t2_ms),
            "--rho-solid-um-per-ms",
            str(params.rho_solid_um_per_ms),
            "--rho-gas-um-per-ms",
            str(params.rho_gas_um_per_ms),
            "--dt-ms",
            str(params.dt_ms),
            "--t-max-ms",
            str(params.t_max_ms),
            "--max-grid-size",
            str(params.max_grid_size),
            "--solver",
            "triangular",
            "--mesh-bulk-size-um",
            str(params.mesh_bulk_size_um),
            "--mesh-boundary-size-um",
            str(params.mesh_boundary_size_um),
            "--mesh-max-points",
            str(params.mesh_max_points),
        ]
        start = time.perf_counter()
        subprocess.run(command, check=True)
        timings[step] = time.perf_counter() - start
    return timings


def compute_comsol_modified_time_deltas(steps: list[dict]) -> dict[int, float]:
    """Use adjacent COMSOL xlsx modification-time differences as solve time.

    For timestep N>1, solve_time_s = mtime(T2_tNNNN.xlsx) - mtime(T2_tNNNN-1.xlsx).
    Timestep 1 has no previous COMSOL file, so it is recorded as NaN.
    """

    timings: dict[int, float] = {}
    previous_mtime: float | None = None
    for item in sorted(steps, key=lambda row: row["timestep"]):
        step = item["timestep"]
        mtime = item["comsol_xlsx"].stat().st_mtime
        timings[step] = math.nan if previous_mtime is None else max(0.0, mtime - previous_mtime)
        previous_mtime = mtime
    return timings


def make_three_way_frames(
    steps: list[dict],
    exp_dir: Path,
    output_dir: Path,
    pixel_decay_dir: Path,
    triangular_decay_dir: Path,
) -> tuple[list[dict], Path]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    time_meta = read_time_metadata(exp_dir)
    summary_rows: list[dict] = []
    first_static_png: Path | None = None

    for frame_idx, item in enumerate(steps, start=1):
        step = item["timestep"]
        info(f"plot three-way comparison timestep {step:04d}")
        pixel_csv = pixel_decay_dir / f"timestep_{step:04d}_nmr_decay.csv"
        tri_csv = triangular_decay_dir / f"timestep_{step:04d}_nmr_decay.csv"

        comsol_t_raw, comsol_s_raw = read_comsol_probe_excel(item["comsol_xlsx"])
        pixel_t_raw, pixel_s_raw = read_png_decay_csv(pixel_csv)
        tri_t_raw, tri_s_raw = read_png_decay_csv(tri_csv)

        comsol_t, comsol_s, comsol_peak = trim_to_decay(comsol_t_raw, comsol_s_raw)
        pixel_t, pixel_s, pixel_peak = trim_to_decay(pixel_t_raw, pixel_s_raw)
        tri_t, tri_s, tri_peak = trim_to_decay(tri_t_raw, tri_s_raw)

        comsol_inv = invert_fixed_alpha(
            comsol_t,
            comsol_s,
            name=f"COMSOL_t{step:04d}",
            alpha=NNLS_ALPHA,
            t2_min_ms=T2_MIN_MS,
            t2_max_ms=T2_MAX_MS,
            num_bins=T2_NUM_BINS,
        )
        pixel_inv = invert_fixed_alpha(
            pixel_t,
            pixel_s,
            name=f"PNG_pixel_t{step:04d}",
            alpha=NNLS_ALPHA,
            t2_min_ms=T2_MIN_MS,
            t2_max_ms=T2_MAX_MS,
            num_bins=T2_NUM_BINS,
        )
        tri_inv = invert_fixed_alpha(
            tri_t,
            tri_s,
            name=f"PNG_triangular_t{step:04d}",
            alpha=NNLS_ALPHA,
            t2_min_ms=T2_MIN_MS,
            t2_max_ms=T2_MAX_MS,
            num_bins=T2_NUM_BINS,
        )

        comsol_spec = comsol_inv.spectrum / max(np.max(comsol_inv.spectrum), 1e-30)
        pixel_spec = pixel_inv.spectrum / max(np.max(pixel_inv.spectrum), 1e-30)
        tri_spec = tri_inv.spectrum / max(np.max(tri_inv.spectrum), 1e-30)

        if frame_idx == 1:
            pd.DataFrame(
                {
                    "t2_ms": comsol_inv.t2_bins_ms,
                    "comsol_spectrum_norm": comsol_spec,
                    "png_pixel_spectrum_norm": pixel_spec,
                    "png_triangular_mesh_spectrum_norm": tri_spec,
                }
            ).to_csv(output_dir / f"timestep_{step:04d}_t2_spectrum_three_way.csv", index=False)
            pd.DataFrame(
                {
                    "comsol_time_ms": pd.Series(comsol_t),
                    "comsol_signal_norm": pd.Series(normalize(comsol_s)),
                    "png_pixel_time_ms": pd.Series(pixel_t),
                    "png_pixel_signal_norm": pd.Series(normalize(pixel_s)),
                    "png_triangular_time_ms": pd.Series(tri_t),
                    "png_triangular_signal_norm": pd.Series(normalize(tri_s)),
                }
            ).to_csv(output_dir / f"timestep_{step:04d}_decay_three_way.csv", index=False)

        meta = time_meta.get(step, {})
        physical_time_s = float(meta["time_s"]) if meta.get("time_s") else math.nan
        porosity = float(meta["porosity"]) if meta.get("porosity") else math.nan
        kk0 = float(meta["k_k0"]) if meta.get("k_k0") else math.nan

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(15.8, 8.6),
            dpi=160,
            gridspec_kw={"width_ratios": [1.08, 1.0]},
        )
        ax_img = axes[:, 0]
        ax_decay = axes[0, 1]
        ax_t2 = axes[1, 1]
        axes[1, 0].remove()
        ax_img = axes[0, 0]
        ax_img.set_position([0.035, 0.10, 0.46, 0.78])

        image = Image.open(item["concentration_png"]).convert("RGB")
        ax_img.imshow(image)
        ax_img.set_axis_off()
        ax_img.set_title("Dissolution / concentration field", fontsize=14)

        ax_decay.plot(comsol_t, normalize(comsol_s), color="#1f77b4", lw=2.0, label="COMSOL")
        ax_decay.plot(pixel_t, normalize(pixel_s), color="#d62728", lw=1.8, ls="--", label="PNG pixel")
        ax_decay.plot(tri_t, normalize(tri_s), color="#2ca02c", lw=1.8, ls="-.", label="PNG triangular mesh")
        ax_decay.set_xlabel("elapsed time after decay peak (ms)")
        ax_decay.set_ylabel("normalized signal")
        ax_decay.set_title("T2 relaxation decay")
        ax_decay.grid(alpha=0.3)
        ax_decay.legend(fontsize=9)

        ax_t2.plot(comsol_inv.t2_bins_ms, comsol_spec, color="#1f77b4", lw=2.0, label="COMSOL")
        ax_t2.plot(pixel_inv.t2_bins_ms, pixel_spec, color="#d62728", lw=1.8, ls="--", label="PNG pixel")
        ax_t2.plot(tri_inv.t2_bins_ms, tri_spec, color="#2ca02c", lw=1.8, ls="-.", label="PNG triangular mesh")
        ax_t2.set_xscale("log")
        ax_t2.set_xlim(T2_MIN_MS, T2_MAX_MS)
        ax_t2.set_xlabel("T2 (ms)")
        ax_t2.set_ylabel("normalized spectral amplitude")
        ax_t2.set_title(f"T2 inversion spectra, alpha={NNLS_ALPHA:g}")
        ax_t2.grid(alpha=0.3, which="both")
        ax_t2.legend(fontsize=9)

        title = f"{exp_dir.name} timestep {step:04d}"
        if np.isfinite(physical_time_s):
            title += f" | time = {physical_time_s:.6g} s"
        if np.isfinite(porosity):
            title += f" | porosity = {porosity:.4f}"
        if np.isfinite(kk0):
            title += f" | k/k0 = {kk0:.3g}"
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(left=0.035, right=0.98, top=0.91, bottom=0.08, wspace=0.28, hspace=0.32)

        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=160, bbox_inches="tight")
        if frame_idx == 1:
            first_static_png = output_dir / f"timestep_{step:04d}_three_way_decay_t2_comparison.png"
            fig.savefig(first_static_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append(
            {
                "timestep": step,
                "time_s": physical_time_s,
                "porosity": porosity,
                "k_k0": kk0,
                "comsol_peak_t2_ms": float(comsol_inv.t2_bins_ms[int(np.argmax(comsol_inv.spectrum))]),
                "png_pixel_peak_t2_ms": float(pixel_inv.t2_bins_ms[int(np.argmax(pixel_inv.spectrum))]),
                "png_triangular_peak_t2_ms": float(tri_inv.t2_bins_ms[int(np.argmax(tri_inv.spectrum))]),
                "comsol_peak_time_raw_ms": float(comsol_t_raw[comsol_peak]),
                "png_pixel_peak_time_raw_ms": float(pixel_t_raw[pixel_peak]),
                "png_triangular_peak_time_raw_ms": float(tri_t_raw[tri_peak]),
            }
        )

    if first_static_png is None:
        raise RuntimeError("No comparison frame was created.")
    return summary_rows, first_static_png


def make_video(frames_dir: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def make_timing_outputs(
    steps: list[dict],
    output_dir: Path,
    comsol_timing: dict[int, float],
    pixel_timing: dict[int, float],
    triangular_timing: dict[int, float],
) -> tuple[Path, Path]:
    rows = []
    for item in steps:
        step = item["timestep"]
        rows.append(
            {
                "timestep": step,
                "comsol_modified_time_delta_s": comsol_timing.get(step, math.nan),
                "png_pixel_wall_time_s": pixel_timing.get(step, math.nan),
                "png_triangular_wall_time_s": triangular_timing.get(step, math.nan),
            }
        )
    timing_csv = output_dir / "solve_time_by_method.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(timing_csv, index=False)

    def mean_label(column: str, label: str) -> str:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            return f"{label} (mean=n/a)"
        return f"{label} (mean={float(np.mean(values)):.3g} s)"

    fig = plt.figure(figsize=(10.5, 10.0), dpi=180)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.05, 1.0], hspace=0.38)
    ax_line = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_box = fig.add_subplot(gs[2, 0])

    series = [
        ("comsol_modified_time_delta_s", "COMSOL xlsx mtime delta", "o", "#1f77b4"),
        ("png_pixel_wall_time_s", "PNG pixel wall time", "s", "#d62728"),
        ("png_triangular_wall_time_s", "PNG triangular mesh wall time", "^", "#2ca02c"),
    ]
    for column, label, marker, color in series:
        ax_line.plot(
            frame["timestep"],
            frame[column],
            marker=marker,
            ms=3,
            lw=1.5,
            color=color,
            label=mean_label(column, label),
        )
    ax_line.set_xlabel("timestep")
    ax_line.set_ylabel("workflow wall time (s)")
    ax_line.set_title("Per-timestep wall time, full scale")
    ax_line.grid(alpha=0.3)
    ax_line.legend(fontsize=8.5)

    zoom_series = [
        ("png_pixel_wall_time_s", "PNG pixel wall time", "s", "#d62728"),
        ("png_triangular_wall_time_s", "PNG triangular mesh wall time", "^", "#2ca02c"),
    ]
    for column, label, marker, color in zoom_series:
        ax_zoom.plot(
            frame["timestep"],
            frame[column],
            marker=marker,
            ms=3,
            lw=1.6,
            color=color,
            label=mean_label(column, label),
        )
    zoom_values = frame[[column for column, *_ in zoom_series]].to_numpy(dtype=float).ravel()
    zoom_values = zoom_values[np.isfinite(zoom_values)]
    if zoom_values.size:
        lo, hi = np.percentile(zoom_values, [1, 99])
        pad = max((hi - lo) * 0.18, 0.2)
        ax_zoom.set_ylim(max(0.0, lo - pad), hi + pad)
    ax_zoom.set_xlabel("timestep")
    ax_zoom.set_ylabel("workflow wall time (s)")
    ax_zoom.set_title("Zoomed comparison without COMSOL")
    ax_zoom.grid(alpha=0.3)
    ax_zoom.legend(fontsize=8.5)

    box_data = [
        frame["comsol_modified_time_delta_s"].dropna().to_numpy(),
        frame["png_pixel_wall_time_s"].dropna().to_numpy(),
        frame["png_triangular_wall_time_s"].dropna().to_numpy(),
    ]
    try:
        ax_box.boxplot(box_data, tick_labels=["COMSOL", "PNG pixel", "PNG mesh"], showmeans=True)
    except TypeError:
        ax_box.boxplot(box_data, labels=["COMSOL", "PNG pixel", "PNG mesh"], showmeans=True)
    ax_box.set_ylabel("workflow wall time (s)")
    ax_box.set_title("Wall-time distribution by method")
    ax_box.grid(alpha=0.3, axis="y")

    timing_png = output_dir / "solve_time_line_and_boxplot.png"
    fig.savefig(timing_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "solve_time_line_boxplot_with_noncomsol_zoom.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return timing_csv, timing_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "exp_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_EXP_DIR,
        help=(
            "RTSPHEM experiment directory, e.g. ...\\exp_015. "
            "If omitted, DEFAULT_EXP_DIR defined near the top of this script is used."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional testing limit. Omit this for a full production run.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = exp_dir / f"three_way_nmr_workflow_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = read_metadata(exp_dir)
    pixel_size_x_um, pixel_size_y_um, geometry_info = compute_pixel_size_from_first_png(exp_dir, metadata)
    steps = collect_aligned_steps(exp_dir, args.max_steps)

    pixel_decay_dir = output_dir / "png_pixel_decay"
    triangular_decay_dir = output_dir / "png_triangular_decay"

    pixel_params = build_params(pixel_size_x_um, pixel_size_y_um, solver="pixel")
    triangular_params = build_params(pixel_size_x_um, pixel_size_y_um, solver="triangular")

    info(f"experiment: {exp_dir}")
    info(f"output: {output_dir}")
    info(f"aligned timesteps: {len(steps)}")
    info(f"raw pixel size: dx={pixel_size_x_um:.6g} um, dy={pixel_size_y_um:.6g} um")

    comsol_timing = compute_comsol_modified_time_deltas(steps)
    pixel_timing = run_pixel_decay(steps, pixel_decay_dir, pixel_params)
    if CURRENT_PYTHON_HAS_PYGIMLI:
        triangular_timing = run_triangular_decay_direct(steps, triangular_decay_dir, triangular_params)
        triangular_runner = sys.executable
    else:
        triangular_timing = run_triangular_decay_subprocess(
            steps,
            triangular_decay_dir,
            triangular_params,
            PYGIMLI_FALLBACK_PYTHON,
        )
        triangular_runner = str(PYGIMLI_FALLBACK_PYTHON)

    summary_rows, first_static_png = make_three_way_frames(
        steps,
        exp_dir,
        output_dir,
        pixel_decay_dir,
        triangular_decay_dir,
    )
    summary_csv = output_dir / "three_way_decay_t2_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    video_path = output_dir / f"{exp_dir.name}_three_way_decay_t2_with_concentration_{alpha_token(NNLS_ALPHA)}.mp4"
    make_video(output_dir / "frames", video_path)
    timing_csv, timing_png = make_timing_outputs(
        steps,
        output_dir,
        comsol_timing,
        pixel_timing,
        triangular_timing,
    )

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "geometry_info": geometry_info,
        "params": {
            "pixel": asdict(pixel_params),
            "triangular": asdict(triangular_params),
            "nnls": {
                "alpha": NNLS_ALPHA,
                "t2_min_ms": T2_MIN_MS,
                "t2_max_ms": T2_MAX_MS,
                "num_bins": T2_NUM_BINS,
            },
            "video_fps": VIDEO_FPS,
            "triangular_runner": triangular_runner,
        },
        "outputs": {
            "video": str(video_path),
            "first_static_comparison_png": str(first_static_png),
            "summary_csv": str(summary_csv),
            "timing_csv": str(timing_csv),
            "timing_png": str(timing_png),
            "png_pixel_decay_dir": str(pixel_decay_dir),
            "png_triangular_decay_dir": str(triangular_decay_dir),
        },
    }
    manifest_path = output_dir / "workflow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    info("done")
    info(f"video: {video_path}")
    info(f"timing plot: {timing_png}")
    info(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
