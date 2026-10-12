"""Overlay the complete Sample 16 grouped simulations and experiment on one T2 axis."""

from __future__ import annotations

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
RUNS = ROOT / "simulation_outputs"
OUTPUT = RUNS / "sample_16_all_nmr_t2_alpha500_20260916"
ALPHA = 500.0


def main() -> None:
    inventory = pd.read_csv(RUNS / "sample_16_rho2_sweep_vs_avizo_20260916" / "rho2_run_inventory_and_peaks.csv")
    config = NnlsConfig(num_bins=200, regularization=ALPHA, t2_min_ms=0.01, t2_max_ms=100000.0)
    rows = []
    peaks = []
    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(inventory)))

    for (_, run), color in zip(inventory.iterrows(), colors):
        manifest_path = Path(run["source_manifest"])
        decay_path = Path(run["source_decay"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["num_completed_groups"] != 70 or manifest["total_weighted_slices"] != 800 or manifest["skipped_group_ids"]:
            raise ValueError(f"Incomplete simulation: {manifest_path}")
        decay = pd.read_csv(decay_path)
        result = invert_single_signal_nnls(
            decay["time_ms"].to_numpy(float),
            decay["group_weighted_normalized_signal"].to_numpy(float),
            signal_name=f"rho2_{run['rho2_um_per_s']:g}", config=config,
        )
        amplitude = np.asarray(result.spectrum, float)
        amplitude /= amplitude.max()
        t2 = np.asarray(result.t2_bins_ms, float)
        rho2 = float(run["rho2_um_per_s"])
        ax.plot(t2, amplitude, color=color, lw=1.8, alpha=0.85, label=rf"Simulation $\rho_2$={rho2:.3f} $\mu$m/s")
        rows.extend({"series": "simulation", "rho2_um_per_s": rho2, "t2_ms": t, "normalized_amplitude": a,
                     "source": str(decay_path.resolve())} for t, a in zip(t2, amplitude))
        peaks.append({"series": "simulation", "rho2_um_per_s": rho2, "t2_peak_ms": float(t2[amplitude.argmax()]),
                      "source": str(decay_path.resolve())})

    experiment_path = Path(manifest["experiment_decay"])
    time, signal, _ = load_experiment_decay(experiment_path)
    result = invert_single_signal_nnls(time, signal, signal_name="sample16_experiment", config=config)
    amplitude = np.asarray(result.spectrum, float)
    amplitude /= amplitude.max()
    t2 = np.asarray(result.t2_bins_ms, float)
    ax.plot(t2, amplitude, color="#cf3b3b", lw=2.8, label="Experiment", zorder=10)
    rows.extend({"series": "experiment", "rho2_um_per_s": np.nan, "t2_ms": t, "normalized_amplitude": a,
                 "source": str(experiment_path.resolve())} for t, a in zip(t2, amplitude))
    peaks.append({"series": "experiment", "rho2_um_per_s": np.nan, "t2_peak_ms": float(t2[amplitude.argmax()]),
                  "source": str(experiment_path.resolve())})

    ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
           ylabel="Normalized T2 amplitude", title=r"Sample 16: experimental and grouped-2D simulated $T_2$ spectra ($\alpha=500$)")
    ax.grid(which="both", alpha=0.16)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = OUTPUT / "sample16_all_nmr_t2_alpha500.png"
    fig.savefig(figure, dpi=300, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUTPUT / "t2_spectra_alpha500.csv", index=False)
    pd.DataFrame(peaks).to_csv(OUTPUT / "t2_peak_summary_alpha500.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "figure": str(figure.resolve()), "inversion_config": config.__dict__,
        "method": "Existing complete 70-group/800-slice weighted decays and the original experimental decay were reinverted with the same fixed-alpha NNLS; each spectrum was normalized to its own maximum.",
        "source_inventory": str((RUNS / "sample_16_rho2_sweep_vs_avizo_20260916" / "rho2_run_inventory_and_peaks.csv").resolve()),
        "experimental_decay": str(experiment_path.resolve()),
        "geometry_note": "Grouped representative 2D slices approximate the 3D CT stack; no common diameter axis because rho2 varies across simulations.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(peaks)[["series", "rho2_um_per_s", "t2_peak_ms"]].to_string(index=False))
    print(figure.resolve())


if __name__ == "__main__":
    main()
