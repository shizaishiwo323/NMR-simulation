"""Compare fixed-alpha rho2=20.742 simulations with the L-curve experiment."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "simulation_outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=int, default=[100, 200])
    args = parser.parse_args()
    if any(alpha <= 0 for alpha in args.alphas) or len(set(args.alphas)) != len(args.alphas):
        raise ValueError("Alphas must be positive and distinct")
    token = "_".join(str(alpha) for alpha in args.alphas)
    output = RUNS / f"sample_16_rho2_20p742_sim_alpha{token}_vs_experiment_20260916"
    source = RUNS / "sample_16_porosity_grouped_nmr_rho2_20p742_peak200_20260914" / "group_weighted_normalized_decay.csv"
    decay = pd.read_csv(source)
    experimental = pd.read_csv(RUNS / "sample_16_rho2_20p742_vs_experiment_lcurve_20260916" / "t2_spectra_pair.csv")
    experimental = experimental[experimental.series == "experiment"].copy()
    if len(experimental) != 200:
        raise ValueError("Expected the 200-bin L-curve experimental spectrum")

    fig, ax = plt.subplots(figsize=(8.4, 5))
    spectra = []
    colors = plt.cm.viridis(np.linspace(0.2, 0.75, len(args.alphas)))
    for alpha, color in zip(args.alphas, colors):
        result = invert_single_signal_nnls(
            decay.time_ms.to_numpy(float), decay.group_weighted_normalized_signal.to_numpy(float),
            signal_name=f"sample16_rho2_20p742_alpha_{alpha}",
            config=NnlsConfig(num_bins=200, regularization=alpha, t2_min_ms=0.01, t2_max_ms=100000.0),
        )
        spectrum = pd.DataFrame({"series": "simulation", "rho2_um_per_s": 20.7419027891008,
                                 "alpha": alpha, "t2_ms": result.t2_bins_ms,
                                 "normalized_amplitude": result.spectrum / result.spectrum.max(),
                                 "source": str(source.resolve())})
        spectra.append(spectrum)
        ax.plot(spectrum.t2_ms, spectrum.normalized_amplitude, color=color, lw=2.1,
                label=rf"Simulation $\rho_2=20.742$ $\mu$m/s ($\alpha={alpha}$)")
    ax.plot(experimental.t2_ms, experimental.normalized_amplitude, color="#cf3d43", lw=2.7,
            label=rf"Experiment, L-curve ($\alpha={experimental.alpha.iloc[0]:.3g}$)")
    ax.set(xscale="log", xlim=(0.01, 100000), ylim=(0, 1.05), xlabel=r"$T_2$ (ms)",
           ylabel="Normalized T2 amplitude", title="Sample 16: simulation smoothing comparison vs experiment")
    ax.grid(which="both", alpha=0.17)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"sample16_rho2_20p742_sim_alpha{token}_vs_experiment.png", dpi=300, facecolor="white")
    plt.close(fig)
    pd.concat([*spectra, experimental], ignore_index=True).to_csv(output / "t2_spectra_comparison.csv", index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()
