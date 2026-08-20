"""L-curve inversion of a completed grouped-slice NMR decay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_t2.config import LCurveConfig
from nmr_t2.lcurve import invert_single_signal_lcurve


DEFAULT_RUN_DIR = Path("simulation_outputs/niu2020_berea_porosity_grouped_nmr_mesh96_alpha0p1_quality0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fixed-spectrum", default="porosity_group_weighted_t2_inversion_fixed_alpha_0p1.csv")
    parser.add_argument("--sample-name", default="Berea grouped mesh-96")
    parser.add_argument("--alpha-min", type=float, default=1e-6)
    parser.add_argument("--alpha-max", type=float, default=1e2)
    parser.add_argument("--alpha-count", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.run_dir / "lcurve_inversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    decay = pd.read_csv(args.run_dir / "group_weighted_normalized_decay.csv")
    config = LCurveConfig(
        num_bins=200,
        t2_min_ms=1e-2,
        t2_max_ms=1e5,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_count=args.alpha_count,
    )
    result = invert_single_signal_lcurve(
        decay["time_ms"].to_numpy(float),
        decay["group_weighted_normalized_signal"].to_numpy(float),
        signal_name=f"{args.sample_name} decay",
        config=config,
    )

    normalized = result.spectrum / max(float(result.spectrum.max()), 1e-30)
    pd.DataFrame(
        {"t2_ms": result.t2_bins_ms, "amplitude": result.spectrum, "normalized_amplitude": normalized}
    ).to_csv(out_dir / "lcurve_optimal_t2_spectrum.csv", index=False)
    metrics = pd.DataFrame(
        {
            "alpha": result.alpha_values,
            "residual_norm": result.residual_norms,
            "roughness_norm": result.roughness_norms,
            "reciprocal_slope": result.slope_reciprocal_values,
            "is_selected": np.arange(result.alpha_values.size) == result.best_index,
        }
    )
    metrics.to_csv(out_dir / "lcurve_metrics.csv", index=False)
    pd.DataFrame(
        {"time_ms": result.fit_time_ms, "fit_amplitude": result.fit_amplitude, "residual": result.residual}
    ).to_csv(out_dir / "lcurve_optimal_fit.csv", index=False)

    fixed_path = args.run_dir / args.fixed_spectrum
    fixed = pd.read_csv(fixed_path)
    summary = {
        "best_alpha": float(result.best_regularization),
        "best_index": int(result.best_index),
        "selected_reciprocal_slope": float(result.slope_reciprocal_values[result.best_index]),
        "used_valid_range_filter": bool(result.used_range_filter),
        "peak_t2_ms": float(result.t2_bins_ms[np.argmax(result.spectrum)]),
        "alpha_scan": {"min": config.alpha_min, "max": config.alpha_max, "count": config.alpha_count},
        "selected_at_scan_boundary": bool(result.best_index in {0, result.alpha_values.size - 1}),
        "selection_method": "closest reciprocal L-curve slope to 0.25 within [0.1, 10]",
        "run_dir": str(args.run_dir.resolve()),
        "fixed_spectrum": str(fixed_path.resolve()),
    }
    (out_dir / "lcurve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, (ax_curve, ax_t2) = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    ax_curve.plot(result.residual_norms, result.roughness_norms, "o-", ms=3, lw=1.2, color="#60758f")
    ax_curve.scatter(
        result.residual_norms[result.best_index], result.roughness_norms[result.best_index],
        s=55, color="#c44e3b", zorder=3, label=rf"selected $\alpha$ = {result.best_regularization:.3g}",
    )
    ax_curve.set_xscale("log")
    ax_curve.set_yscale("log")
    residual_ticks = np.geomspace(result.residual_norms.min(), result.residual_norms.max(), 4)
    ax_curve.set_xticks(residual_ticks, [f"{value:.2f}" for value in residual_ticks])
    ax_curve.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_curve.set_xlabel("Residual norm")
    ax_curve.set_ylabel("Roughness norm")
    ax_curve.set_title("a  L-curve selection", loc="left", fontweight="bold")
    ax_curve.legend()

    ax_t2.plot(fixed["t2_ms"], fixed["normalized_amplitude"], color="#9ca3af", lw=1.5, label=r"fixed $\alpha=0.1$")
    ax_t2.plot(result.t2_bins_ms, normalized, color="#1f2937", lw=2.0, label=rf"L-curve $\alpha={result.best_regularization:.3g}$")
    ax_t2.set_xscale("log")
    ax_t2.set_xlim(100, 3000)
    ax_t2.set_xlabel(r"$T_2$ (ms)")
    ax_t2.set_ylabel("Normalized amplitude")
    ax_t2.set_title("b  T2 inversion", loc="left", fontweight="bold")
    ax_t2.legend()
    fig.suptitle(f"{args.sample_name} regularization selection", fontsize=10, fontweight="bold")
    fig.savefig(out_dir / "lcurve_selection_and_t2_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
