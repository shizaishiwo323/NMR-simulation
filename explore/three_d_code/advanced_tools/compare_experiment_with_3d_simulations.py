"""Compare experimental CPMG decay with multiple 3D NMR simulations.

The experimental CSV is read without a header and only its first two columns
are used. Each simulation decay CSV must contain ``time_ms`` plus either
``normalized_signal``, ``average_normalized_signal``, or ``signal``. All curves
are normalized by their first valid point and inverted with the same fixed-alpha
NNLS settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_EXPERIMENT_CSV = Path(
    r"C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\sip模拟"
    r"\data_inventory\ct_backed_samples_raw_copy_20260605\sample_89_Grainstone"
    r"\NMR_combined\89\T2CPMG\data.csv"
)


@dataclass(frozen=True)
class DecaySeries:
    name: str
    path: Path
    time_ms: np.ndarray
    normalized_signal: np.ndarray


def alpha_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def normalize_decay(time_ms: np.ndarray, signal: np.ndarray, *, min_time_positive: bool) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(time_ms) & np.isfinite(signal)
    if min_time_positive:
        valid &= time_ms > 0
    time = np.asarray(time_ms[valid], dtype=float)
    values = np.asarray(signal[valid], dtype=float)
    order = np.argsort(time)
    time = time[order]
    values = values[order]
    if time.size < 2:
        raise ValueError("Decay series has fewer than two valid points.")
    first = float(values[0])
    if abs(first) < 1e-30:
        raise ValueError("First valid signal value is zero; cannot normalize.")
    return time, values / first


def load_experiment_decay(path: Path) -> DecaySeries:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"Experimental CSV must contain at least two columns: {path}")
    time = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    time, normalized = normalize_decay(time, signal, min_time_positive=True)
    return DecaySeries("Experiment", path, time, normalized)


def load_simulation_decay(name: str, path: Path) -> DecaySeries:
    raw = pd.read_csv(path)
    if "time_ms" not in raw.columns:
        raise ValueError(f"Simulation decay CSV must contain 'time_ms': {path}")
    time = pd.to_numeric(raw["time_ms"], errors="coerce").to_numpy(dtype=float)
    if "normalized_signal" in raw.columns:
        signal_column = "normalized_signal"
    elif "average_normalized_signal" in raw.columns:
        signal_column = "average_normalized_signal"
    elif "signal" in raw.columns:
        signal_column = "signal"
    else:
        raise ValueError(
            f"Simulation decay CSV must contain normalized_signal, average_normalized_signal, or signal: {path}"
        )
    signal = pd.to_numeric(raw[signal_column], errors="coerce").to_numpy(dtype=float)
    time, normalized = normalize_decay(time, signal, min_time_positive=True)
    return DecaySeries(name, path, time, normalized)


def parse_simulation_arg(value: str) -> tuple[str, Path]:
    if "|" not in value:
        raise argparse.ArgumentTypeError("Simulation entries must use the form 'label|path'.")
    label, raw_path = value.split("|", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise argparse.ArgumentTypeError("Simulation label cannot be empty.")
    return label, path


def save_decay_long(series: list[DecaySeries], output_path: Path) -> None:
    rows = []
    for item in series:
        rows.append(
            pd.DataFrame(
                {
                    "series": item.name,
                    "source_path": str(item.path.resolve()),
                    "time_ms": item.time_ms,
                    "normalized_signal": item.normalized_signal,
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_spectrum(result, source_name: str, output_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "series": source_name,
            "t2_ms": result.t2_bins_ms,
            "amplitude": result.spectrum,
            "normalized_amplitude": result.spectrum / max(float(np.max(result.spectrum)), 1e-30),
        }
    )
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def save_comparison_figure(
    series: list[DecaySeries],
    spectra: list[pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.9))
    colors = {
        "Experiment": "#0b6e69",
        "Full CPU 3D": "#111827",
        "pyGIMLi tetra 150": "#b45309",
    }
    max_sim_time = max(float(item.time_ms.max()) for item in series if item.name != "Experiment")
    for item in series:
        frame = pd.DataFrame({"time_ms": item.time_ms, "normalized_signal": item.normalized_signal})
        if item.name == "Experiment":
            frame = frame[frame["time_ms"] <= max_sim_time]
        axs[0].plot(
            frame["time_ms"],
            frame["normalized_signal"],
            lw=1.8 if item.name == "Experiment" else 2.2,
            color=colors.get(item.name),
            label=item.name,
        )
    axs[0].set_xlabel("time (ms)")
    axs[0].set_ylabel("normalized signal")
    axs[0].set_title("Decay comparison")
    axs[0].grid(alpha=0.28)
    axs[0].legend()

    for spectrum in spectra:
        name = str(spectrum["series"].iloc[0])
        axs[1].plot(
            spectrum["t2_ms"],
            spectrum["normalized_amplitude"],
            lw=1.8 if name == "Experiment" else 2.2,
            color=colors.get(name),
            label=name,
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
    parser.add_argument(
        "--simulation",
        action="append",
        type=parse_simulation_arg,
        required=True,
        help="Simulation decay entry as 'label|path'. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_89_3d_experiment_comparison"))
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not args.experiment_csv.exists():
        raise FileNotFoundError(f"Experimental CSV not found: {args.experiment_csv}")
    for _, sim_path in args.simulation:
        if not sim_path.exists():
            raise FileNotFoundError(f"Simulation decay CSV not found: {sim_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = NnlsConfig(
        num_bins=int(args.t2_bins),
        regularization=float(args.fixed_alpha),
        t2_min_ms=float(args.t2_min_ms),
        t2_max_ms=float(args.t2_max_ms),
    )
    experiment = load_experiment_decay(args.experiment_csv)
    simulations = [load_simulation_decay(label, path) for label, path in args.simulation]
    all_series = [experiment, *simulations]

    decay_long_path = args.output_dir / "experiment_and_3d_simulations_decay_long.csv"
    save_decay_long(all_series, decay_long_path)

    spectra = []
    spectrum_outputs = {}
    for item in all_series:
        result = invert_single_signal_nnls(item.time_ms, item.normalized_signal, signal_name=item.name, config=cfg)
        file_token = item.name.lower().replace(" ", "_").replace("/", "_")
        spectrum_path = args.output_dir / f"{file_token}_t2_inversion_fixed_alpha_{alpha_token(args.fixed_alpha)}.csv"
        spectra.append(save_spectrum(result, item.name, spectrum_path))
        spectrum_outputs[item.name] = str(spectrum_path.resolve())

    spectra_long_path = args.output_dir / "experiment_and_3d_simulations_t2_spectra_long.csv"
    pd.concat(spectra, ignore_index=True).to_csv(spectra_long_path, index=False, encoding="utf-8-sig")
    figure_path = args.output_dir / "experiment_vs_3d_simulations_decay_and_t2.png"
    save_comparison_figure(all_series, spectra, figure_path)

    manifest = {
        "experiment_csv": str(args.experiment_csv.resolve()),
        "experiment_columns_used": [0, 1],
        "simulations": [{"label": label, "decay_csv": str(path.resolve())} for label, path in args.simulation],
        "output_dir": str(args.output_dir.resolve()),
        "inversion_config": asdict(cfg),
        "series_points": {item.name: int(item.time_ms.size) for item in all_series},
        "series_time_range_ms": {
            item.name: [float(item.time_ms.min()), float(item.time_ms.max())] for item in all_series
        },
        "outputs": {
            "decay_long_csv": str(decay_long_path.resolve()),
            "spectra_long_csv": str(spectra_long_path.resolve()),
            "comparison_figure_png": str(figure_path.resolve()),
            "individual_spectrum_csv": spectrum_outputs,
        },
        "comparison_note": (
            "Experimental and simulated decays were normalized by their first valid point. "
            "All T2 inversions used the same fixed-alpha NNLS configuration."
        ),
    }
    manifest_path = args.output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparison outputs written to: {args.output_dir.resolve()}")
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
