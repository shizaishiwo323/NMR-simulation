from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT.parent
    / "SIP模拟"
    / "sip模拟"
    / "results"
    / "pore_network"
    / "5-16seged"
    / "pnextract_network"
    / "network_parsed"
)
OUTPUT = (
    ROOT
    / "simulation_outputs"
    / "sample_16_2D"
    / "sample16_pnextract_pore_throat_frequency_histograms.png"
)


def read_column(path: Path, name: str) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return np.asarray([float(row[name]) for row in csv.DictReader(handle)])


def main() -> None:
    pore_diameter_um = 2e6 * read_column(DATA_DIR / "pores.csv", "pore_radius_m")
    throat_length_um = 1e6 * read_column(DATA_DIR / "throats.csv", "throat_length_m")

    assert pore_diameter_um.size == 6815
    assert throat_length_um.size == 11081
    assert np.all(np.isfinite(pore_diameter_um)) and np.all(pore_diameter_um > 0)
    assert np.all(np.isfinite(throat_length_um)) and np.all(throat_length_um > 0)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    panels = [
        (pore_diameter_um, "Pore diameter distribution", "Pore diameter ($\mu$m)"),
        (throat_length_um, "Pore throat length distribution", "Pore throat length ($\mu$m)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    for label, ax, (values, title, xlabel) in zip("ab", axes, panels):
        edges = np.geomspace(values.min(), values.max(), 49)
        counts, _ = np.histogram(values, bins=edges)
        ax.bar(
            edges[:-1],
            counts * 100.0 / counts.sum(),
            width=np.diff(edges),
            align="edge",
            color="#8A8A8A",
            edgecolor="#202020",
            linewidth=0.65,
        )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency fraction (%)")
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", top=True, right=True)
        ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
    print(f"Pores: {pore_diameter_um.size}; throat lengths: {throat_length_um.size}")


if __name__ == "__main__":
    main()
