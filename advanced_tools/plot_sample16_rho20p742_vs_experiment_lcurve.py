"""Show the existing L-curve T2 spectra for rho2=20.742 and the experiment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_outputs" / "sample_16_all_nmr_individual_lcurve_20260916"
OUTPUT = ROOT / "simulation_outputs" / "sample_16_rho2_20p742_vs_experiment_lcurve_20260916"
RHO2 = 20.7419027891008


def main() -> None:
    spectra = pd.read_csv(SOURCE / "t2_spectra_individual_lcurve.csv")
    selected = spectra[(spectra.series == "experiment") |
                       ((spectra.series == "simulation") & np.isclose(spectra.rho2_um_per_s, RHO2))].copy()
    if len(selected) != 400:
        raise ValueError("Expected one 200-bin simulated spectrum and one 200-bin experimental spectrum")

    fig, ax = plt.subplots(figsize=(8.4, 5))
    for series, color, width in (("simulation", "#277a9d", 2.2), ("experiment", "#cf3d43", 2.7)):
        row = selected[selected.series == series]
        alpha = float(row.alpha.iloc[0])
        label = (rf"Simulation $\rho_2=20.742$ $\mu$m/s ($\alpha={alpha:.3g}$)" if series == "simulation"
                 else rf"Experiment ($\alpha={alpha:.3g}$)")
        ax.plot(row.t2_ms, row.normalized_amplitude, color=color, lw=width, label=label)
    ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
           ylabel="Normalized T2 amplitude", title="Sample 16: grouped-2D simulation vs experiment")
    ax.grid(which="both", alpha=0.17)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "sample16_rho2_20p742_vs_experiment_lcurve.png", dpi=300, facecolor="white")
    plt.close(fig)
    selected.to_csv(OUTPUT / "t2_spectra_pair.csv", index=False)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
