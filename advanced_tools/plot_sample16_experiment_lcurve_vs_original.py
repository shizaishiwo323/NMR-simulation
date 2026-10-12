"""Compare Sample 16 experimental L-curve inversion with the vendor T2 spectrum."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.compare_experimental_t2_with_sim_average import load_experiment_decay
from nmr_t2.config import LCurveConfig
from nmr_t2.lcurve import invert_single_signal_lcurve


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "simulation_outputs" / "sample_16_experiment_lcurve_vs_original_20260916"


def main() -> None:
    previous = ROOT / "simulation_outputs" / "sample_16_peak300_rho2_31p113_exp500_sim500_20260914" / "run_manifest.json"
    inputs = json.loads(previous.read_text(encoding="utf-8"))["inputs"]
    decay_path = Path(inputs["experimental_decay"])
    vendor_path = Path(inputs["experimental_spectrum"])
    time, signal, _ = load_experiment_decay(decay_path)
    config = LCurveConfig(num_bins=200, t2_min_ms=0.01, t2_max_ms=100000.0,
                          alpha_min=1e-6, alpha_max=1e6, alpha_count=100)
    result = invert_single_signal_lcurve(time, signal, signal_name="Sample 16 experiment", config=config)
    if result.best_index in (0, len(result.alpha_values) - 1):
        raise ValueError(f"Selected alpha {result.best_regularization:g} lies on the scan boundary; expand the range.")

    vendor = pd.read_csv(vendor_path, header=None, names=["t2_ms", "amplitude"])
    vendor = vendor.replace([np.inf, -np.inf], np.nan).dropna()
    vendor = vendor[(vendor.t2_ms > 0) & (vendor.amplitude >= 0)].copy()
    vendor["normalized_amplitude"] = vendor.amplitude / vendor.amplitude.max()
    inverted = pd.DataFrame({"t2_ms": result.t2_bins_ms, "amplitude": result.spectrum,
                             "normalized_amplitude": result.spectrum / result.spectrum.max()})
    metrics = pd.DataFrame({"alpha": result.alpha_values, "residual_norm": result.residual_norms,
                            "roughness_norm": result.roughness_norms, "reciprocal_slope": result.slope_reciprocal_values,
                            "selected": np.arange(len(result.alpha_values)) == result.best_index})

    fig, (ax_l, ax_t2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax_l.loglog(result.residual_norms, result.roughness_norms, "o-", color="#587792", ms=3, lw=1)
    ax_l.scatter(result.residual_norms[result.best_index], result.roughness_norms[result.best_index],
                 color="#d1493f", s=65, zorder=5, label=rf"Selected $\alpha={result.best_regularization:.3g}$")
    ax_l.set(xlabel="Residual norm", ylabel="Roughness norm", title="L-curve selection")
    ax_l.grid(which="both", alpha=0.18)
    ax_l.legend(frameon=False)

    ax_t2.plot(vendor.t2_ms, vendor.normalized_amplitude, color="#26374a", lw=2.1,
               label="Original experimental spectrum")
    ax_t2.plot(inverted.t2_ms, inverted.normalized_amplitude, color="#d1493f", lw=2.1,
               label=rf"L-curve reinversion ($\alpha={result.best_regularization:.3g}$)")
    ax_t2.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
              ylabel="Normalized T2 amplitude", title="Experimental T2 spectrum")
    ax_t2.grid(which="both", alpha=0.18)
    ax_t2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = OUTPUT / "sample16_experiment_lcurve_vs_original.png"
    fig.savefig(figure, dpi=300, facecolor="white")
    plt.close(fig)

    metrics.to_csv(OUTPUT / "lcurve_metrics.csv", index=False)
    inverted.to_csv(OUTPUT / "lcurve_reinverted_t2_spectrum.csv", index=False)
    vendor.to_csv(OUTPUT / "original_experimental_t2_spectrum.csv", index=False)
    summary = {
        "experimental_decay": str(decay_path.resolve()), "original_spectrum": str(vendor_path.resolve()),
        "inversion_config": config.__dict__, "selection_method": "Project reciprocal L-curve slope closest to 0.25 within [0.1, 10]",
        "best_alpha": float(result.best_regularization), "selected_reciprocal_slope": float(result.slope_reciprocal_values[result.best_index]),
        "used_range_filter": bool(result.used_range_filter),
        "vendor_peak_t2_ms": float(vendor.t2_ms.iloc[vendor.amplitude.argmax()]),
        "reinverted_peak_t2_ms": float(inverted.t2_ms.iloc[inverted.amplitude.argmax()]),
        "normalization": "Each spectrum divided by its own maximum for shape comparison",
        "figure": str(figure.resolve()),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
