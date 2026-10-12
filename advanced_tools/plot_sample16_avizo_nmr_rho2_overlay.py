"""Plot Avizo CT volumes with experimental and grouped-2D NMR diameter maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histogram", type=Path, required=True)
    parser.add_argument("--experiment-spectrum", type=Path, required=True)
    parser.add_argument("--simulation-spectrum", type=Path, required=True)
    parser.add_argument("--rho2-um-per-s", type=float, required=True)
    parser.add_argument("--simulation-alpha", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--diameter-max-um", type=float, default=1500.0)
    return parser.parse_args()


def mapped_spectrum(frame: pd.DataFrame, rho2_um_per_s: float) -> pd.DataFrame:
    result = frame[["t2_ms", "amplitude"]].copy()
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result = result[(result["t2_ms"] > 0) & (result["amplitude"] >= 0)]
    if result.empty or result["amplitude"].max() <= 0:
        raise ValueError("Spectrum needs positive T2 values and a nonzero amplitude.")
    result["normalized_amplitude"] = result["amplitude"] / result["amplitude"].max()
    result["diameter_um"] = 0.006 * rho2_um_per_s * result["t2_ms"]
    return result


def main() -> None:
    args = parse_args()
    hist = pd.read_csv(args.histogram)
    experiment_frame = pd.read_csv(args.experiment_spectrum)
    vendor_experiment = "t2_ms" not in experiment_frame.columns
    if vendor_experiment:
        experiment_frame = pd.read_csv(args.experiment_spectrum, header=None, names=["t2_ms", "amplitude"])
    experiment = mapped_spectrum(experiment_frame, args.rho2_um_per_s)
    simulation = mapped_spectrum(pd.read_csv(args.simulation_spectrum), args.rho2_um_per_s)
    simulation_label = "porosity-grouped 2D (2.5D) NMR"
    if args.simulation_alpha is not None:
        simulation_label += rf" ($\alpha={args.simulation_alpha:g}$)"

    edges = np.r_[hist["diameter_left_um"].to_numpy(float), hist["diameter_right_um"].iloc[-1]]
    pore_volume_um3 = hist["pore_volume_weighted_frequency"].to_numpy(float) * 1e18
    throat_volume_um3 = hist["throat_volume_weighted_frequency"].to_numpy(float) * 1e18
    centers = hist["diameter_center_um"].to_numpy(float)
    pore_peak = centers[np.argmax(pore_volume_um3)]
    throat_peak = centers[np.argmax(throat_volume_um3)]
    exp_peak = experiment.loc[experiment["normalized_amplitude"].idxmax(), "diameter_um"]
    sim_peak = simulation.loc[simulation["normalized_amplitude"].idxmax(), "diameter_um"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    nmr_ax = ax.twinx()
    ax.bar(
        edges[:-1], pore_volume_um3, width=np.diff(edges), align="edge",
        color="#f4a3a8", alpha=0.50, edgecolor="none",
        label=rf"Avizo CT pore body (peak {pore_peak:.0f} $\mu$m)",
    )
    ax.bar(
        edges[:-1], throat_volume_um3, width=np.diff(edges), align="edge",
        color="none", edgecolor="#df6f73", linewidth=0.75, alpha=0.85,
        label=rf"Avizo CT throat (peak {throat_peak:.0f} $\mu$m)",
    )
    nmr_ax.plot(
        experiment["diameter_um"], experiment["normalized_amplitude"],
        color="black", linewidth=1.15, marker="o", markersize=2.7,
        markevery=max(1, len(experiment) // 35),
        label=rf"experimental NMR {'vendor $T_2$ spectrum, ' if vendor_experiment else ''}sphere map (peak {exp_peak:.0f} $\mu$m)",
    )
    nmr_ax.plot(
        simulation["diameter_um"], simulation["normalized_amplitude"],
        color="#555555", linewidth=1.25, linestyle="--",
        label=rf"{simulation_label} (peak {sim_peak:.0f} $\mu$m)",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.02, args.diameter_max_um)
    ax.set_ylim(bottom=0)
    nmr_ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Diameter, $d$ ($\mu$m)")
    ax.set_ylabel(r"CT volume per bin ($\mu$m$^3$)", color="#e5484d")
    nmr_ax.set_ylabel("Normalized NMR amplitude")
    ax.tick_params(axis="y", colors="#e5484d")
    ax.spines["left"].set_color("#e5484d")
    ax.set_title(
        rf"Sample 16: Avizo CT and NMR distributions "
        rf"($\rho_2={args.rho2_um_per_s:.3f}$ $\mu$m s$^{{-1}}$)"
    )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = nmr_ax.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plot_data = pd.concat(
        [
            hist.assign(series="Avizo CT histogram"),
            experiment.assign(series="experimental NMR vendor spectrum" if vendor_experiment else "experimental NMR"),
            simulation.assign(series=f"porosity-grouped 2D (2.5D) NMR; alpha={args.simulation_alpha:g}" if args.simulation_alpha is not None else simulation_label),
        ],
        ignore_index=True,
        sort=False,
    )
    plot_data.to_csv(args.source_data, index=False)
    print(f"Figure: {args.output.resolve()}")
    print(f"Peaks (um): pore={pore_peak:.3f}, throat={throat_peak:.3f}, experiment={exp_peak:.3f}, simulation={sim_peak:.3f}")


if __name__ == "__main__":
    main()
