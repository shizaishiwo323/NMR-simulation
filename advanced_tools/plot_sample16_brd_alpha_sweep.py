"""Plot T2process MATLAB BRD alpha sweep against the vendor spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "simulation_outputs" / "sample_16_experiment_brd_alpha_log_20260916"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    output = args.supplement_dir or args.run_dir or OUTPUT
    input_dir = args.run_dir or OUTPUT
    data = np.loadtxt(input_dir / "brd_t2_spectra.csv", delimiter=",")
    summary = pd.read_csv(input_dir / "brd_summary.csv", header=None,
                          names=["alpha", "t2_peak_ms", "relative_residual", "min_amplitude"])
    if args.supplement_dir:
        extra = np.loadtxt(output / "brd_t2_spectra.csv", delimiter=",")
        extra_summary = pd.read_csv(output / "brd_summary.csv", header=None, names=summary.columns)
        assert np.allclose(data[:, 0], extra[:, 0]) and extra.shape == (200, 3)
        data = np.column_stack([extra, data[:, 1:]])
        summary = pd.concat([extra_summary, summary], ignore_index=True)
    assert data.shape == (200, len(summary) + 1)
    t2 = data[:, 0]
    spectra = data[:, 1:]
    normalized = spectra / spectra.max(axis=0)
    assert np.all(np.isfinite(normalized)) and np.all(normalized >= -1e-10)

    prior = ROOT / "simulation_outputs" / "sample_16_experiment_lcurve_vs_original_20260916" / "manifest.json"
    vendor_path = Path(json.loads(prior.read_text(encoding="utf-8"))["original_spectrum"])
    vendor = pd.read_csv(vendor_path, header=None, names=["t2_ms", "amplitude"])
    vendor = vendor.replace([np.inf, -np.inf], np.nan).dropna()
    vendor = vendor[(vendor.t2_ms > 0) & (vendor.amplitude >= 0)].copy()

    fig, (ax, map_ax) = plt.subplots(1, 2, figsize=(13.4, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(summary)))
    ax.plot(vendor.t2_ms, vendor.amplitude / vendor.amplitude.max(), color="#202d3a", lw=2.7,
            label="Original experimental spectrum", zorder=5)
    for i, (row, color) in enumerate(zip(summary.itertuples(), colors)):
        ax.plot(t2, normalized[:, i], color=color, lw=1.35, alpha=0.9, label=rf"BRD $\alpha={row.alpha:g}$")
    ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
           ylabel="Normalized amplitude", title="MATLAB BRD spectra and original experiment")
    ax.grid(which="both", alpha=0.15)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

    mesh = map_ax.pcolormesh(t2, summary.alpha, normalized.T, shading="auto", cmap="viridis", vmin=0, vmax=1)
    map_ax.plot(summary.t2_peak_ms, summary.alpha, "w.-", lw=1.1, ms=4, label="Peak T2")
    map_ax.set(xscale="log", yscale="linear" if args.run_dir else "log", xlim=(0.01, 100000),
               ylim=(summary.alpha.min(), summary.alpha.max()),
               xlabel=r"$T_2$ (ms)", ylabel=r"BRD smoothing factor $\alpha$",
               title="Spectrum shape across smoothing factors")
    map_ax.legend(loc="lower left", frameon=False, labelcolor="white", fontsize=8)
    fig.colorbar(mesh, ax=map_ax, label="Peak-normalized amplitude")
    fig.tight_layout()
    figure = output / ("sample16_matlab_brd_alpha_0p2_1p0_spectra.png" if args.run_dir else "sample16_matlab_brd_alpha_log_spectra.png")
    fig.savefig(figure, dpi=300, facecolor="white")
    plt.close(fig)

    detail_figure = None
    if args.run_dir:
        detail, detail_ax = plt.subplots(figsize=(8.5, 5.1))
        detail_ax.plot(vendor.t2_ms, vendor.amplitude / vendor.amplitude.max(), color="#202d3a", lw=2.6,
                       label="Original experimental spectrum")
        for i, (row, color) in enumerate(zip(summary.itertuples(), colors)):
            detail_ax.plot(t2, normalized[:, i], color=color, lw=1.6, label=rf"BRD $\alpha={row.alpha:.1f}$")
        detail_ax.set(xscale="log", xlim=(10, 3000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
                      ylabel="Normalized amplitude", title="Sample 16: BRD smoothing factors 0.2 to 1.0")
        detail_ax.grid(which="both", alpha=0.16)
        detail_ax.legend(frameon=False, fontsize=8, ncol=2)
        detail.tight_layout()
        detail_figure = output / "sample16_matlab_brd_alpha_0p2_1p0_detail.png"
        detail.savefig(detail_figure, dpi=300, facecolor="white")
        plt.close(detail)
    if args.supplement_dir:
        detail, detail_ax = plt.subplots(figsize=(8.5, 5.1))
        detail_ax.plot(vendor.t2_ms, vendor.amplitude / vendor.amplitude.max(), color="#202d3a", lw=2.6,
                       label="Original experimental spectrum")
        for target, color in [(0.01, "#3765a0"), (0.1, "#2f9b82"), (1, "#d38734")]:
            i = int(np.flatnonzero(np.isclose(summary.alpha, target))[0])
            detail_ax.plot(t2, normalized[:, i], color=color, lw=2, label=rf"MATLAB BRD $\alpha={target:g}$")
        detail_ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
                      ylabel="Normalized amplitude", title="Sample 16: low-alpha MATLAB BRD comparison")
        detail_ax.grid(which="both", alpha=0.16)
        detail_ax.legend(frameon=False)
        detail.tight_layout()
        detail_figure = output / "sample16_matlab_brd_alpha_0p01_0p1_detail.png"
        detail.savefig(detail_figure, dpi=300, facecolor="white")
        plt.close(detail)

    pd.DataFrame(np.column_stack([t2, normalized]), columns=["t2_ms"] + [f"alpha_{a:g}" for a in summary.alpha]).to_csv(
        output / "brd_normalized_t2_spectra.csv", index=False)
    summary.to_csv(output / "brd_summary_labeled.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "source_repository": "https://github.com/shizaishiwo323/T2process",
        "source_commit": "47461a72e0852e5d2a5a08db19a1f8404145e236",
        "matlab_solver": "brd/matlab/brd1.p", "matlab_sampling": "brd/matlab/AmplitudeSample.m",
        "matlab_version": "R2025b", "driver": str((ROOT / "advanced_tools" / "sample16_brd_alpha_sweep.m").resolve()),
        "experimental_decay": str(Path(json.loads(prior.read_text(encoding="utf-8"))["experimental_decay"]).resolve()),
        "original_spectrum": str(vendor_path.resolve()),
        "sampled_echo_count": 200, "t2_bins": 200, "t2_range_ms": [0.01, 100000],
        "alpha_values": summary.alpha.tolist(), "normalization": "Each spectrum divided by its own maximum",
        "figure": str(figure.resolve()),
        "low_alpha_detail_figure": str(detail_figure.resolve()) if detail_figure else None,
        "prior_sweep": str(OUTPUT.resolve()) if args.supplement_dir else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(figure.resolve())


if __name__ == "__main__":
    main()
