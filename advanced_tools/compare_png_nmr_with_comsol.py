"""Compare PNG-derived NMR decay with a COMSOL T2 export and fixed-alpha inversion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_t2.config import NnlsConfig
from nmr_t2.io_utils import cell_to_float, parse_time_cell
from nmr_t2.nnls import invert_single_signal_nnls


def read_comsol_probe_excel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the first valid COMSOL probe signal from a workbook."""

    table = pd.read_excel(path, header=None, dtype=object)
    times_s: list[float] = []
    signals: list[float] = []
    for row in table.itertuples(index=False):
        ok, time_s = parse_time_cell(row[0])
        if not ok:
            continue
        signal = cell_to_float(row[1]) if len(row) > 1 else np.nan
        if np.isfinite(signal):
            times_s.append(time_s)
            signals.append(signal)

    if not times_s:
        raise ValueError(f"No valid COMSOL time/signal rows found in {path}")

    time_ms = np.asarray(times_s, dtype=float) * 1000.0
    signal = np.asarray(signals, dtype=float)
    valid = np.isfinite(time_ms) & np.isfinite(signal) & (time_ms >= 0)
    time_ms = time_ms[valid]
    signal = signal[valid]
    order = np.argsort(time_ms)
    return time_ms[order], signal[order]


def read_png_decay_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    time_ms = frame["time_ms"].to_numpy(dtype=float)
    signal_col = "signal" if "signal" in frame.columns else "normalized_signal"
    signal = frame[signal_col].to_numpy(dtype=float)
    valid = np.isfinite(time_ms) & np.isfinite(signal) & (time_ms >= 0)
    return time_ms[valid], signal[valid]


def trim_to_decay(time_ms: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Start from the global maximum and reset time to elapsed milliseconds."""

    peak_idx = int(np.argmax(signal))
    trimmed_time = time_ms[peak_idx:] - time_ms[peak_idx]
    trimmed_signal = signal[peak_idx:]
    return trimmed_time, trimmed_signal, peak_idx


def normalize(signal: np.ndarray) -> np.ndarray:
    first = float(signal[0]) if signal.size else 0.0
    return signal / max(abs(first), 1e-30)


def infer_case_name(png_decay_csv: Path) -> str:
    stem = png_decay_csv.stem
    if stem.endswith("_nmr_decay"):
        return stem[: -len("_nmr_decay")]
    return stem


def alpha_token(alpha: float) -> str:
    return f"alpha{alpha:g}".replace(".", "p").replace("-", "m")


def invert_fixed_alpha(
    time_ms: np.ndarray,
    signal: np.ndarray,
    *,
    name: str,
    alpha: float,
    t2_min_ms: float,
    t2_max_ms: float,
    num_bins: int,
):
    config = NnlsConfig(
        num_bins=int(num_bins),
        regularization=float(alpha),
        t2_min_ms=float(t2_min_ms),
        t2_max_ms=float(t2_max_ms),
        min_points_after_trim=10,
    )
    return invert_single_signal_nnls(time_ms, signal, signal_name=name, config=config)


def plot_decay(
    png_time_ms: np.ndarray,
    png_signal: np.ndarray,
    comsol_time_ms: np.ndarray,
    comsol_signal: np.ndarray,
    extra_comsol_time_ms: np.ndarray | None,
    extra_comsol_signal: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(comsol_time_ms, normalize(comsol_signal), color="#1f77b4", lw=2.2, label="COMSOL original workflow")
    if extra_comsol_time_ms is not None and extra_comsol_signal is not None:
        ax.plot(
            extra_comsol_time_ms,
            normalize(extra_comsol_signal),
            color="#2ca02c",
            lw=2.0,
            label="COMSOL manual boundaries",
        )
    ax.plot(png_time_ms, normalize(png_signal), color="#d62728", lw=2.0, ls="--", label="PNG finite-difference workflow")
    ax.set_xlabel("elapsed time after decay peak (ms)")
    ax.set_ylabel("normalized signal")
    ax.set_title("T2 decay comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_t2_spectrum(png_result, comsol_result, extra_comsol_result, output_path: Path) -> None:
    png_spec = png_result.spectrum / max(np.max(png_result.spectrum), 1e-30)
    comsol_spec = comsol_result.spectrum / max(np.max(comsol_result.spectrum), 1e-30)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        comsol_result.t2_bins_ms,
        comsol_spec,
        color="#1f77b4",
        lw=2.2,
        label=f"COMSOL, alpha={comsol_result.regularization:g}",
    )
    if extra_comsol_result is not None:
        extra_spec = extra_comsol_result.spectrum / max(np.max(extra_comsol_result.spectrum), 1e-30)
        ax.plot(
            extra_comsol_result.t2_bins_ms,
            extra_spec,
            color="#2ca02c",
            lw=2.0,
            label=f"COMSOL manual boundaries, alpha={extra_comsol_result.regularization:g}",
        )
    ax.plot(
        png_result.t2_bins_ms,
        png_spec,
        color="#d62728",
        lw=2.0,
        ls="--",
        label=f"PNG workflow, alpha={png_result.regularization:g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("T2 (ms)")
    ax.set_ylabel("normalized spectral amplitude")
    ax.set_title("Fixed-alpha T2 inversion comparison")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-xlsx", type=Path, required=True)
    parser.add_argument("--extra-comsol-xlsx", type=Path, default=None)
    parser.add_argument("--png-decay-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--t2-min-ms", type=float, default=1.0)
    parser.add_argument("--t2-max-ms", type=float, default=1e4)
    parser.add_argument("--num-bins", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comsol_time_raw, comsol_signal_raw = read_comsol_probe_excel(args.comsol_xlsx)
    extra_time_raw = extra_signal_raw = None
    if args.extra_comsol_xlsx is not None:
        extra_time_raw, extra_signal_raw = read_comsol_probe_excel(args.extra_comsol_xlsx)
    png_time_raw, png_signal_raw = read_png_decay_csv(args.png_decay_csv)
    case_name = infer_case_name(args.png_decay_csv)

    comsol_time, comsol_signal, comsol_peak_idx = trim_to_decay(comsol_time_raw, comsol_signal_raw)
    extra_time = extra_signal = None
    extra_peak_idx = None
    if extra_time_raw is not None and extra_signal_raw is not None:
        extra_time, extra_signal, extra_peak_idx = trim_to_decay(extra_time_raw, extra_signal_raw)
    png_time, png_signal, png_peak_idx = trim_to_decay(png_time_raw, png_signal_raw)

    comsol_inv = invert_fixed_alpha(
        comsol_time,
        comsol_signal,
        name="COMSOL_t0002",
        alpha=args.alpha,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
        num_bins=args.num_bins,
    )
    png_inv = invert_fixed_alpha(
        png_time,
        png_signal,
        name="PNG_t0002",
        alpha=args.alpha,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
        num_bins=args.num_bins,
    )
    extra_inv = None
    if extra_time is not None and extra_signal is not None:
        extra_inv = invert_fixed_alpha(
            extra_time,
            extra_signal,
            name="COMSOL_manual_boundaries",
            alpha=args.alpha,
            t2_min_ms=args.t2_min_ms,
            t2_max_ms=args.t2_max_ms,
            num_bins=args.num_bins,
        )

    decay_png = args.output_dir / f"{case_name}_decay_comparison.png"
    alpha_name = alpha_token(float(args.alpha))
    t2_png = args.output_dir / f"{case_name}_t2_inversion_comparison_{alpha_name}.png"
    plot_decay(png_time, png_signal, comsol_time, comsol_signal, extra_time, extra_signal, decay_png)
    plot_t2_spectrum(png_inv, comsol_inv, extra_inv, t2_png)

    pd.DataFrame(
        {
            "time_ms": png_time,
            "png_signal": png_signal,
            "png_normalized": normalize(png_signal),
        }
    ).to_csv(args.output_dir / "png_decay_trimmed.csv", index=False)
    pd.DataFrame(
        {
            "time_ms": comsol_time,
            "comsol_signal": comsol_signal,
            "comsol_normalized": normalize(comsol_signal),
        }
    ).to_csv(args.output_dir / "comsol_decay_trimmed.csv", index=False)
    if extra_time is not None and extra_signal is not None:
        pd.DataFrame(
            {
                "time_ms": extra_time,
                "extra_comsol_signal": extra_signal,
                "extra_comsol_normalized": normalize(extra_signal),
            }
        ).to_csv(args.output_dir / "extra_comsol_decay_trimmed.csv", index=False)
    t2_export = {
        "t2_ms": png_inv.t2_bins_ms,
        "png_spectrum": png_inv.spectrum,
        "png_spectrum_normalized": png_inv.spectrum / max(np.max(png_inv.spectrum), 1e-30),
        "comsol_spectrum": comsol_inv.spectrum,
        "comsol_spectrum_normalized": comsol_inv.spectrum / max(np.max(comsol_inv.spectrum), 1e-30),
    }
    if extra_inv is not None:
        t2_export["extra_comsol_spectrum"] = extra_inv.spectrum
        t2_export["extra_comsol_spectrum_normalized"] = extra_inv.spectrum / max(np.max(extra_inv.spectrum), 1e-30)
    pd.DataFrame(
        t2_export
    ).to_csv(args.output_dir / f"t2_inversion_comparison_{alpha_name}.csv", index=False)

    summary = {
        "comsol_xlsx": str(args.comsol_xlsx.resolve()),
        "extra_comsol_xlsx": str(args.extra_comsol_xlsx.resolve()) if args.extra_comsol_xlsx is not None else None,
        "png_decay_csv": str(args.png_decay_csv.resolve()),
        "decay_comparison_png": str(decay_png.resolve()),
        "t2_comparison_png": str(t2_png.resolve()),
        "inversion_config": asdict(
            NnlsConfig(
                num_bins=int(args.num_bins),
                regularization=float(args.alpha),
                t2_min_ms=float(args.t2_min_ms),
                t2_max_ms=float(args.t2_max_ms),
            )
        ),
        "comsol_peak_time_ms_raw": float(comsol_time_raw[comsol_peak_idx]),
        "extra_comsol_peak_time_ms_raw": (
            float(extra_time_raw[extra_peak_idx])
            if extra_time_raw is not None and extra_peak_idx is not None
            else None
        ),
        "png_peak_time_ms_raw": float(png_time_raw[png_peak_idx]),
        "comsol_decay_points": int(comsol_time.size),
        "extra_comsol_decay_points": int(extra_time.size) if extra_time is not None else None,
        "png_decay_points": int(png_time.size),
        "comsol_residual_norm": float(comsol_inv.residual_norm),
        "extra_comsol_residual_norm": float(extra_inv.residual_norm) if extra_inv is not None else None,
        "png_residual_norm": float(png_inv.residual_norm),
        "comsol_peak_t2_ms": float(comsol_inv.t2_bins_ms[int(np.argmax(comsol_inv.spectrum))]),
        "extra_comsol_peak_t2_ms": (
            float(extra_inv.t2_bins_ms[int(np.argmax(extra_inv.spectrum))]) if extra_inv is not None else None
        ),
        "png_peak_t2_ms": float(png_inv.t2_bins_ms[int(np.argmax(png_inv.spectrum))]),
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Comparison outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
