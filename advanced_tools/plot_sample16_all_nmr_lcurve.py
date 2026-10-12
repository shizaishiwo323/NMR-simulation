"""Overlay Sample 16 simulated and experimental T2 spectra with separate L-curve alpha choices."""

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
RUNS = ROOT / "simulation_outputs"
OUTPUT = RUNS / "sample_16_all_nmr_individual_lcurve_20260916"
CONFIG = LCurveConfig(num_bins=200, t2_min_ms=0.01, t2_max_ms=100000.0,
                      alpha_min=1e-6, alpha_max=1e6, alpha_count=100)


def main() -> None:
    inventory_path = RUNS / "sample_16_rho2_sweep_vs_avizo_20260916" / "rho2_run_inventory_and_peaks.csv"
    inventory = pd.read_csv(inventory_path)
    if len(inventory) != 8:
        raise ValueError(f"Expected eight complete simulations: {inventory_path}")
    sources = []
    for _, run in inventory.iterrows():
        manifest_path = Path(run.source_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["num_completed_groups"] != 70 or manifest["total_weighted_slices"] != 800 or manifest["skipped_group_ids"]:
            raise ValueError(f"Incomplete simulation: {manifest_path}")
        decay_path = Path(run.source_decay)
        decay = pd.read_csv(decay_path)
        sources.append(("simulation", float(run.rho2_um_per_s), decay_path,
                        decay.time_ms.to_numpy(float), decay.group_weighted_normalized_signal.to_numpy(float)))
    experiment_paths = {json.loads(Path(p).read_text(encoding="utf-8"))["experiment_decay"]
                        for p in inventory.source_manifest}
    if len(experiment_paths) != 1:
        raise ValueError("Simulation runs refer to different experimental decays")
    experiment_path = Path(experiment_paths.pop())
    time, signal, _ = load_experiment_decay(experiment_path)
    sources.append(("experiment", np.nan, experiment_path, time, signal))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(inventory)))
    spectra, summaries, metrics = [], [], []
    for index, (series, rho2, path, time, signal) in enumerate(sources):
        result = invert_single_signal_lcurve(time, signal, signal_name=f"sample16_{series}_{rho2:g}", config=CONFIG)
        if result.best_index in (0, len(result.alpha_values) - 1):
            raise ValueError(f"L-curve selection at scan boundary for {path}: alpha={result.best_regularization:g}")
        amplitude = result.spectrum / result.spectrum.max()
        color = "#d13d40" if series == "experiment" else colors[index]
        label = (rf"Experiment ($\alpha={result.best_regularization:.3g}$)" if series == "experiment"
                 else rf"Simulation $\rho_2={rho2:.3f}$ $\mu$m/s ($\alpha={result.best_regularization:.3g}$)")
        ax.plot(result.t2_bins_ms, amplitude, color=color, lw=2.7 if series == "experiment" else 1.75,
                alpha=1 if series == "experiment" else 0.9, label=label,
                zorder=10 if series == "experiment" else 2)
        spectra.extend({"series": series, "rho2_um_per_s": rho2, "alpha": result.best_regularization,
                        "t2_ms": t, "normalized_amplitude": a, "source": str(path.resolve())}
                       for t, a in zip(result.t2_bins_ms, amplitude))
        summaries.append({"series": series, "rho2_um_per_s": rho2, "selected_alpha": result.best_regularization,
                          "selected_reciprocal_slope": result.slope_reciprocal_values[result.best_index],
                          "used_range_filter": result.used_range_filter,
                          "t2_peak_ms": float(result.t2_bins_ms[np.argmax(result.spectrum)]),
                          "source": str(path.resolve())})
        metrics.extend({"series": series, "rho2_um_per_s": rho2, "alpha": alpha,
                        "residual_norm": residual, "roughness_norm": roughness,
                        "reciprocal_slope": slope, "selected": i == result.best_index}
                       for i, (alpha, residual, roughness, slope) in enumerate(zip(
                           result.alpha_values, result.residual_norms, result.roughness_norms,
                           result.slope_reciprocal_values)))

    ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
           ylabel="Normalized T2 amplitude", title="Sample 16: individual L-curve inversions of experiment and grouped-2D simulations")
    ax.grid(which="both", alpha=0.16)
    ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper left")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = OUTPUT / "sample16_all_nmr_individual_lcurve.png"
    fig.savefig(figure, dpi=300, facecolor="white")
    plt.close(fig)
    pd.DataFrame(spectra).to_csv(OUTPUT / "t2_spectra_individual_lcurve.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "lcurve_alpha_and_peak_summary.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT / "lcurve_metrics_all_series.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "figure": str(figure.resolve()), "inversion_config": CONFIG.__dict__,
        "selection_method": "Project reciprocal L-curve slope closest to 0.25 within [0.1, 10]",
        "source_inventory": str(inventory_path.resolve()), "experimental_decay": str(experiment_path.resolve()),
        "normalization": "Each spectrum divided by its own maximum",
        "simulation_method": "Porosity-grouped 2D representative slices approximate the 3D CT stack",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(summaries)[["series", "rho2_um_per_s", "selected_alpha", "t2_peak_ms"]].to_string(index=False))
    print(figure.resolve())


if __name__ == "__main__":
    main()
