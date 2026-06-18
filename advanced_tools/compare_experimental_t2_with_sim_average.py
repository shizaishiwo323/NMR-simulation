"""Compare experimental T2 CPMG data with simulated average-slice T2 inversion.

The experimental CSV is read without a header; only the first two columns are
used as time and decay amplitude. Both experimental and simulated average decay
curves are normalized by their first valid point before fixed-alpha NNLS
inversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_EXPERIMENT_CSV = Path(
    r"C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\sip模拟"
    r"\data_inventory\ct_backed_samples_raw_copy_20260605\sample_89_Grainstone"
    r"\NMR_combined\89\T2CPMG\data.csv"
)
DEFAULT_SIM_AVERAGE_DECAY = Path(
    r"C:\Users\imgw\Documents\Codex\NMR模拟"
    r"\simulation_outputs\sample_89_random10_nmr\average_normalized_decay.csv"
)


def load_experiment_decay(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"Experimental CSV must contain at least two columns: {path}")
    time_ms = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_ms) & np.isfinite(signal) & (time_ms > 0)
    time_ms = time_ms[valid]
    signal = signal[valid]
    order = np.argsort(time_ms)
    time_ms = time_ms[order]
    signal = signal[order]
    if time_ms.size < 2:
        raise ValueError("Experimental decay has fewer than two valid points.")
    normalized = signal / max(float(signal[0]), 1e-30)
    frame = pd.DataFrame({"time_ms": time_ms, "signal": signal, "normalized_signal": normalized})
    return time_ms, normalized, frame


def load_sim_average_decay(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw = pd.read_csv(path)
    time_ms = pd.to_numeric(raw["time_ms"], errors="coerce").to_numpy(dtype=float)
    if "average_normalized_signal" in raw.columns:
        signal_column = "average_normalized_signal"
    elif "normalized_signal" in raw.columns:
        signal_column = "normalized_signal"
    else:
        raise ValueError(
            "Simulated decay CSV must contain either 'average_normalized_signal' "
            "or 'normalized_signal'."
        )
    signal = pd.to_numeric(raw[signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_ms) & np.isfinite(signal) & (time_ms > 0)
    time_ms = time_ms[valid]
    signal = signal[valid]
    order = np.argsort(time_ms)
    time_ms = time_ms[order]
    signal = signal[order]
    if time_ms.size < 2:
        raise ValueError("Simulated average decay has fewer than two valid points.")
    normalized = signal / max(float(signal[0]), 1e-30)
    frame = pd.DataFrame({"time_ms": time_ms, "normalized_signal": normalized})
    return time_ms, normalized, frame


def save_inversion(result, output_path: Path) -> None:
    pd.DataFrame(
        {
            "t2_ms": result.t2_bins_ms,
            "amplitude": result.spectrum,
            "normalized_amplitude": result.spectrum / max(float(np.max(result.spectrum)), 1e-30),
        }
    ).to_csv(output_path, index=False)


def save_comparison_figure(
    exp_decay: pd.DataFrame,
    sim_decay: pd.DataFrame,
    exp_result,
    sim_result,
    output_path: Path,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8))

    sim_t_max = float(sim_decay["time_ms"].max())
    exp_for_decay = exp_decay[exp_decay["time_ms"] <= sim_t_max]
    axs[0].plot(
        exp_for_decay["time_ms"],
        exp_for_decay["normalized_signal"],
        color="#0b6e69",
        lw=1.8,
        label="experiment",
    )
    axs[0].plot(
        sim_decay["time_ms"],
        sim_decay["normalized_signal"],
        color="#1f2937",
        lw=2.2,
        label="simulation average",
    )
    axs[0].set_xlabel("time (ms)")
    axs[0].set_ylabel("normalized signal")
    axs[0].set_title("Decay comparison")
    axs[0].grid(alpha=0.28)
    axs[0].legend()

    axs[1].plot(
        exp_result.t2_bins_ms,
        exp_result.spectrum / max(float(np.max(exp_result.spectrum)), 1e-30),
        color="#0b6e69",
        lw=2.2,
        label="experiment",
    )
    axs[1].plot(
        sim_result.t2_bins_ms,
        sim_result.spectrum / max(float(np.max(sim_result.spectrum)), 1e-30),
        color="#1f2937",
        lw=2.2,
        label="simulation average",
    )
    axs[1].set_xscale("log")
    axs[1].set_xlabel("T2 (ms)")
    axs[1].set_ylabel("normalized amplitude")
    axs[1].set_title("Fixed-alpha T2 inversion")
    axs[1].grid(alpha=0.28, which="both")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-csv", type=Path, default=DEFAULT_EXPERIMENT_CSV)
    parser.add_argument("--sim-average-decay", type=Path, default=DEFAULT_SIM_AVERAGE_DECAY)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_89_random10_nmr/experiment_comparison"))
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.experiment_csv.exists():
        raise FileNotFoundError(f"Experimental CSV not found: {args.experiment_csv}")
    if not args.sim_average_decay.exists():
        raise FileNotFoundError(f"Simulated average decay CSV not found: {args.sim_average_decay}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = NnlsConfig(
        num_bins=args.t2_bins,
        regularization=args.fixed_alpha,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
    )
    exp_time, exp_signal, exp_decay_frame = load_experiment_decay(args.experiment_csv)
    sim_time, sim_signal, sim_decay_frame = load_sim_average_decay(args.sim_average_decay)

    exp_result = invert_single_signal_nnls(exp_time, exp_signal, signal_name="experiment_T2CPMG", config=cfg)
    sim_result = invert_single_signal_nnls(sim_time, sim_signal, signal_name="simulation_average_10_slices", config=cfg)

    exp_decay_path = args.output_dir / "experiment_decay_first_two_columns_normalized.csv"
    sim_decay_path = args.output_dir / "simulation_average_decay_for_comparison.csv"
    exp_spectrum_path = args.output_dir / "experiment_t2_inversion_fixed_alpha_476p4.csv"
    sim_spectrum_path = args.output_dir / "simulation_average_t2_reinversion_fixed_alpha_476p4.csv"
    figure_path = args.output_dir / "experiment_vs_sim_average_decay_and_t2_comparison.png"
    manifest_path = args.output_dir / "comparison_manifest.json"

    exp_decay_frame.to_csv(exp_decay_path, index=False)
    sim_decay_frame.to_csv(sim_decay_path, index=False)
    save_inversion(exp_result, exp_spectrum_path)
    save_inversion(sim_result, sim_spectrum_path)
    save_comparison_figure(exp_decay_frame, sim_decay_frame, exp_result, sim_result, figure_path)

    manifest = {
        "experiment_csv": str(args.experiment_csv.resolve()),
        "experiment_columns_used": [0, 1],
        "sim_average_decay_csv": str(args.sim_average_decay.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "inversion_config": cfg.__dict__,
        "experiment_points_used": int(exp_time.size),
        "simulation_points_used": int(sim_time.size),
        "experiment_time_range_ms": [float(exp_time.min()), float(exp_time.max())],
        "simulation_time_range_ms": [float(sim_time.min()), float(sim_time.max())],
        "outputs": {
            "experiment_decay_csv": str(exp_decay_path.resolve()),
            "simulation_decay_csv": str(sim_decay_path.resolve()),
            "experiment_spectrum_csv": str(exp_spectrum_path.resolve()),
            "simulation_spectrum_csv": str(sim_spectrum_path.resolve()),
            "comparison_figure_png": str(figure_path.resolve()),
        },
        "comparison_note": (
            "Experimental and simulated average decays were normalized by their first valid point. "
            "Both inversions used the same fixed-alpha NNLS parameters."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparison outputs written to: {args.output_dir.resolve()}")
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
