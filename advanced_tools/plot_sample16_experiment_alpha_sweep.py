"""Compare four fixed-alpha reinversions of Sample 16 experimental decay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.compare_experimental_t2_with_sim_average import load_experiment_decay
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHAS = (2000, 2500, 3000, 3500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs=4, type=int, default=DEFAULT_ALPHAS)
    args = parser.parse_args()
    alphas = tuple(args.alphas)
    output = ROOT / "simulation_outputs" / f"sample_16_experiment_alpha_{min(alphas)}_{max(alphas)}_20260916"
    source = ROOT / "simulation_outputs" / "sample_16_experiment_lcurve_vs_original_20260916" / "manifest.json"
    inputs = json.loads(source.read_text(encoding="utf-8"))
    decay_path = Path(inputs["experimental_decay"])
    vendor_path = Path(inputs["original_spectrum"])
    time, signal, _ = load_experiment_decay(decay_path)
    vendor = pd.read_csv(vendor_path, header=None, names=["t2_ms", "amplitude"])
    vendor = vendor.replace([np.inf, -np.inf], np.nan).dropna()
    vendor = vendor[(vendor.t2_ms > 0) & (vendor.amplitude >= 0)].copy()
    vendor["normalized_amplitude"] = vendor.amplitude / vendor.amplitude.max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax in axes:
        ax.plot(vendor.t2_ms, vendor.normalized_amplitude, color="#26374a", lw=2.3,
                label="Original experimental spectrum")
    spectra = []
    peaks = [{"series": "original experimental spectrum", "alpha": np.nan,
              "t2_peak_ms": float(vendor.t2_ms.iloc[vendor.amplitude.argmax()])}]
    colors = ("#4277ad", "#36a19a", "#d89732", "#c94b50")
    for alpha, color in zip(alphas, colors):
        config = NnlsConfig(num_bins=200, regularization=alpha, t2_min_ms=0.01, t2_max_ms=100000.0)
        result = invert_single_signal_nnls(time, signal, signal_name=f"sample16_experiment_alpha_{alpha}", config=config)
        spectrum = pd.DataFrame({"alpha": alpha, "t2_ms": result.t2_bins_ms,
                                 "amplitude": result.spectrum,
                                 "normalized_amplitude": result.spectrum / result.spectrum.max()})
        spectra.append(spectrum)
        peaks.append({"series": "reinverted experiment", "alpha": alpha,
                      "t2_peak_ms": float(spectrum.t2_ms.iloc[spectrum.amplitude.argmax()])})
        for ax in axes:
            ax.plot(spectrum.t2_ms, spectrum.normalized_amplitude, color=color, lw=1.9,
                    label=rf"Reinversion $\alpha={alpha}$")

    axes[0].set_title("Full T2 range")
    axes[0].set_xlim(0.01, 100000)
    axes[1].set_title("Main-peak detail")
    axes[1].set_xlim(100, 3000)
    for ax in axes:
        ax.set(xscale="log", ylim=(0, 1.05), xlabel=r"$T_2$ (ms)", ylabel="Normalized T2 amplitude")
        ax.grid(which="both", alpha=0.16)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure = output / f"sample16_experiment_alpha_{min(alphas)}_{max(alphas)}_vs_original.png"
    fig.savefig(figure, dpi=300, facecolor="white")
    plt.close(fig)
    pd.concat(spectra, ignore_index=True).to_csv(output / "fixed_alpha_t2_spectra.csv", index=False)
    pd.DataFrame(peaks).to_csv(output / "t2_peak_summary.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experimental_decay": str(decay_path.resolve()), "original_spectrum": str(vendor_path.resolve()),
        "alphas": alphas, "t2_bins": 200, "t2_range_ms": [0.01, 100000.0],
        "normalization": "Each spectrum divided by its own maximum for shape comparison",
        "figure": str(figure.resolve()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(peaks).to_string(index=False))
    print(figure.resolve())


if __name__ == "__main__":
    main()
