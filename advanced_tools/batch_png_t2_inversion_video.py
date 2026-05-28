"""Batch invert PNG-derived NMR decay curves and render a time-labeled video."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import savemat


def add_t2_process_to_path(path: Path) -> None:
    sys.path.insert(0, str(path))


def timestep_from_name(path: Path) -> int:
    match = re.search(r"timestep_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse timestep from {path.name}")
    return int(match.group(1))


def normalize(values: np.ndarray) -> np.ndarray:
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value <= 0 or not np.isfinite(max_value):
        return np.zeros_like(values, dtype=float)
    return values / max_value


def read_times(path: Path) -> dict[int, float]:
    frame = pd.read_csv(path)
    if "timestep" not in frame.columns or "time_s" not in frame.columns:
        raise ValueError(f"{path} must contain timestep and time_s columns.")
    return {int(row.timestep): float(row.time_s) for row in frame.itertuples(index=False)}


def plot_frame(
    *,
    png_path: Path,
    time_ms: np.ndarray,
    signal: np.ndarray,
    t2_ms: np.ndarray,
    spectrum: np.ndarray,
    timestep: int,
    physical_time_s: float | None,
    peak_t2_ms: float,
    output_path: Path,
) -> None:
    image = np.asarray(Image.open(png_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[0].set_title("phase map")

    axes[1].plot(time_ms, signal / max(float(signal[0]), 1e-30), color="black", lw=1.8)
    axes[1].set_xlabel("echo time (ms)")
    axes[1].set_ylabel("normalized signal")
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("T2 decay")

    axes[2].plot(t2_ms, normalize(spectrum), color="#d62728", lw=2.0)
    axes[2].set_xscale("log")
    axes[2].set_xlim(float(t2_ms[0]), float(t2_ms[-1]))
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].set_xlabel("T2 (ms)")
    axes[2].set_ylabel("normalized amplitude")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].set_title(f"NNLS spectrum, peak={peak_t2_ms:.1f} ms")

    time_text = "time unavailable" if physical_time_s is None else f"RTM time = {physical_time_s:g} s"
    fig.suptitle(f"exp_013 timestep {timestep:04d} | {time_text} | alpha=0.5", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decay-dir", required=True, type=Path)
    parser.add_argument("--interface-dir", required=True, type=Path)
    parser.add_argument("--time-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--t2-process-path", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num-bins", type=int, default=200)
    parser.add_argument("--t2-min-ms", type=float, default=1.0)
    parser.add_argument("--t2-max-ms", type=float, default=1e4)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    add_t2_process_to_path(args.t2_process_path)
    from nmr_t2.config import NnlsConfig
    from nmr_t2.nnls import invert_single_signal_nnls

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir = args.output_dir / "spectra_csv"
    figures_dir = args.output_dir / "inversion_figures"
    frames_dir = args.output_dir / "video_frames"
    for path in (spectra_dir, figures_dir, frames_dir):
        path.mkdir(parents=True, exist_ok=True)

    time_by_step = read_times(args.time_csv)
    decay_paths = sorted(args.decay_dir.glob("timestep_*_nmr_decay.csv"), key=timestep_from_name)
    if not decay_paths:
        raise FileNotFoundError(f"No timestep_*_nmr_decay.csv files found in {args.decay_dir}")

    config = NnlsConfig(
        num_bins=int(args.num_bins),
        regularization=float(args.alpha),
        t2_min_ms=float(args.t2_min_ms),
        t2_max_ms=float(args.t2_max_ms),
        min_points_after_trim=10,
    )

    summary_rows = []
    frame_paths = []
    for decay_path in decay_paths:
        timestep = timestep_from_name(decay_path)
        decay = pd.read_csv(decay_path)
        time_ms = decay["time_ms"].to_numpy(dtype=float)
        signal = decay["signal"].to_numpy(dtype=float)
        result = invert_single_signal_nnls(time_ms, signal, signal_name=f"timestep_{timestep:04d}", config=config)

        spectrum_csv = spectra_dir / f"timestep_{timestep:04d}_t2_spectrum_alpha0p5.csv"
        pd.DataFrame(
            {
                "t2_ms": result.t2_bins_ms,
                "spectrum": result.spectrum,
                "spectrum_normalized": normalize(result.spectrum),
            }
        ).to_csv(spectrum_csv, index=False)

        mat_path = spectra_dir / f"timestep_{timestep:04d}_t2_spectrum_alpha0p5.mat"
        savemat(
            mat_path,
            {
                "T2_bins_ms": result.t2_bins_ms.reshape(-1, 1),
                "spectrum": result.spectrum.reshape(-1, 1),
                "regularization": np.array([[float(args.alpha)]], dtype=float),
                "time_ms": time_ms.reshape(-1, 1),
                "signal": signal.reshape(-1, 1),
            },
            do_compression=True,
        )

        peak_idx = int(np.nanargmax(result.spectrum)) if result.spectrum.size else 0
        peak_t2_ms = float(result.t2_bins_ms[peak_idx]) if result.t2_bins_ms.size else float("nan")
        physical_time = time_by_step.get(timestep)
        png_path = args.interface_dir / f"timestep_{timestep:04d}.png"

        fig_path = figures_dir / f"timestep_{timestep:04d}_t2_inversion_alpha0p5.png"
        frame_path = frames_dir / f"frame_{timestep:04d}.png"
        plot_frame(
            png_path=png_path,
            time_ms=time_ms,
            signal=signal,
            t2_ms=result.t2_bins_ms,
            spectrum=result.spectrum,
            timestep=timestep,
            physical_time_s=physical_time,
            peak_t2_ms=peak_t2_ms,
            output_path=fig_path,
        )
        plot_frame(
            png_path=png_path,
            time_ms=time_ms,
            signal=signal,
            t2_ms=result.t2_bins_ms,
            spectrum=result.spectrum,
            timestep=timestep,
            physical_time_s=physical_time,
            peak_t2_ms=peak_t2_ms,
            output_path=frame_path,
        )
        frame_paths.append(frame_path)

        summary_rows.append(
            {
                "timestep": timestep,
                "time_s": physical_time,
                "decay_csv": str(decay_path.resolve()),
                "spectrum_csv": str(spectrum_csv.resolve()),
                "spectrum_mat": str(mat_path.resolve()),
                "figure_png": str(fig_path.resolve()),
                "initial_signal": float(signal[0]),
                "final_normalized_signal": float(signal[-1] / max(signal[0], 1e-30)),
                "peak_t2_ms": peak_t2_ms,
                "residual_norm": float(result.residual_norm),
                "roughness_norm": float(result.roughness_norm),
                "regularization": float(args.alpha),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = args.output_dir / "batch_t2_inversion_summary_alpha0p5.csv"
    summary.to_csv(summary_csv, index=False)

    video_path = args.output_dir / "exp_013_t2_inversion_alpha0p5.mp4"
    gif_path = args.output_dir / "exp_013_t2_inversion_alpha0p5.gif"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(float(args.fps)),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        video_output = video_path
    except Exception:
        images = [imageio.imread(frame_path) for frame_path in frame_paths]
        imageio.mimsave(gif_path, images, duration=1.0 / float(args.fps))
        video_output = gif_path

    manifest = {
        "decay_dir": str(args.decay_dir.resolve()),
        "interface_dir": str(args.interface_dir.resolve()),
        "time_csv": str(args.time_csv.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "alpha": float(args.alpha),
        "num_steps": len(summary_rows),
        "summary_csv": str(summary_csv.resolve()),
        "video": str(video_output.resolve()),
        "video_mp4": str(video_path.resolve()) if video_path.exists() else None,
        "video_gif": str(gif_path.resolve()) if gif_path.exists() else None,
    }
    (args.output_dir / "batch_t2_inversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
