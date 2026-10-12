"""Overlay MATLAB BRD alpha=1 spectra for Sample 16 simulations and experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "simulation_outputs"
OUTPUT = RUNS / "sample_16_all_nmr_matlab_brd_alpha1_20260916"
INVENTORY = RUNS / "sample_16_rho2_sweep_vs_avizo_20260916" / "rho2_run_inventory_and_peaks.csv"


def main() -> None:
    inventory = pd.read_csv(INVENTORY)
    assert len(inventory) == 8
    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(inventory)))
    spectra = []
    peaks = []
    for i, (run, color) in enumerate(zip(inventory.itertuples(), colors), start=1):
        path = OUTPUT / f"simulation_{i:02d}" / "brd_t2_spectra.csv"
        data = np.loadtxt(path, delimiter=",")
        summary = np.loadtxt(path.parent / "brd_summary.csv", delimiter=",")
        assert data.shape == (200, 2) and np.isclose(summary[0], 1)
        t2, amplitude = data.T
        normalized = amplitude / amplitude.max()
        rho2 = float(run.rho2_um_per_s)
        ax.plot(t2, normalized, color=color, lw=1.8, alpha=0.9,
                label=rf"Simulation $\rho_2={rho2:.3f}$ $\mu$m/s")
        spectra.extend({"series": "simulation", "rho2_um_per_s": rho2, "t2_ms": t,
                        "normalized_amplitude": a, "source_decay": run.source_decay}
                       for t, a in zip(t2, normalized))
        peaks.append({"series": "simulation", "rho2_um_per_s": rho2, "t2_peak_ms": summary[1],
                      "relative_residual": summary[2], "first_echo_ms": 5.0,
                      "source_decay": run.source_decay})

    experiment_dir = OUTPUT / "experiment"
    data = np.loadtxt(experiment_dir / "brd_t2_spectra.csv", delimiter=",")
    summary = np.loadtxt(experiment_dir / "brd_summary.csv", delimiter=",")
    assert data.shape == (200, 2) and np.isclose(summary[0], 1)
    t2, amplitude = data.T
    normalized = amplitude / amplitude.max()
    experiment_path = json.loads((RUNS / "sample_16_experiment_lcurve_vs_original_20260916" / "manifest.json").read_text(encoding="utf-8"))["experimental_decay"]
    ax.plot(t2, normalized, color="#cf3b3b", lw=2.8, label="Experiment (BRD)", zorder=10)
    spectra.extend({"series": "experiment", "rho2_um_per_s": np.nan, "t2_ms": t,
                    "normalized_amplitude": a, "source_decay": experiment_path}
                   for t, a in zip(t2, normalized))
    peaks.append({"series": "experiment", "rho2_um_per_s": np.nan, "t2_peak_ms": summary[1],
                  "relative_residual": summary[2], "first_echo_ms": 0.6,
                  "source_decay": experiment_path})

    ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
           ylabel="Normalized T2 amplitude", title=r"Sample 16: experimental and grouped-2D simulated $T_2$ spectra (MATLAB BRD $\alpha=1$)")
    ax.grid(which="both", alpha=0.16)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    figure = OUTPUT / "sample16_all_nmr_matlab_brd_alpha1.png"
    fig.savefig(figure, dpi=300, facecolor="white")
    plt.close(fig)
    pd.DataFrame(spectra).to_csv(OUTPUT / "t2_spectra_brd_alpha1.csv", index=False)
    pd.DataFrame(peaks).to_csv(OUTPUT / "t2_peak_and_fit_summary_brd_alpha1.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "figure": str(figure.resolve()), "method": "MATLAB T2process brd1.p, alpha=1, 200 AmplitudeSample echoes, 200 log-spaced T2 bins from 0.01 to 100000 ms",
        "source_repository_commit": "https://github.com/shizaishiwo323/T2process/tree/47461a72e0852e5d2a5a08db19a1f8404145e236/brd/matlab",
        "source_inventory": str(INVENTORY.resolve()), "experimental_decay": experiment_path,
        "input_normalization": "Each decay divided by its first valid measured echo before BRD inversion",
        "display_normalization": "Each BRD spectrum divided by its own maximum",
        "first_echo_ms": {"simulation": 5.0, "experiment": 0.6},
        "simulation_geometry": "70 grouped 2D representative slices weighted across 800 CT slices; 2.5D approximation, not a true 3D solve",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(peaks)[["series", "rho2_um_per_s", "t2_peak_ms", "relative_residual"]].to_string(index=False))
    print(figure.resolve())


if __name__ == "__main__":
    main()
