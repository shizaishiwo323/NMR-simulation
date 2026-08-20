"""Combine slice-90 mesh-density sensitivity results into one figure and tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("simulation_outputs")
OUT = ROOT / "niu2020_berea_slice0090_mesh_sensitivity"
CASES = {
    32: ROOT / "niu2020_berea_slice0090_mesh32_sensitivity",
    48: ROOT / "niu2020_berea_slice0090_mesh48_sensitivity",
    64: ROOT / "niu2020_berea_slice0090_mesh64_sensitivity",
    80: ROOT / "niu2020_berea_slice0090_mesh80_sensitivity",
    96: ROOT / "niu2020_berea_slice0090_dense96_probe",
    128: ROOT / "niu2020_berea_slice0090_dense128_probe",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spectra: dict[int, pd.DataFrame] = {}
    rows = []

    for grid, directory in CASES.items():
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        item = manifest["per_slice_summaries"][0]
        mesh = item["mesh_or_boundary_summary"]
        spectrum = pd.read_csv(item["inversion_outputs"]["spectrum_csv"])
        spectra[grid] = spectrum
        amplitude = spectrum["amplitude"].to_numpy(float)
        t2 = spectrum["t2_ms"].to_numpy(float)
        weights = amplitude / amplitude.sum()
        rows.append(
            {
                "max_grid_size": grid,
                "simulation_pixels": int(np.prod(item["simulation_shape"])),
                "mesh_nodes": mesh["mesh_nodes"],
                "mesh_triangles": mesh["mesh_triangles"],
                "mean_element_quality": mesh["mean_element_quality"],
                "peak_t2_ms": t2[np.argmax(amplitude)],
                "log_weighted_mean_t2_ms": np.exp(np.sum(weights * np.log(t2))),
                "total_amplitude": amplitude.sum(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("max_grid_size").reset_index(drop=True)
    reference = spectra[128]["amplitude"].to_numpy(float).copy()
    reference /= reference.sum()
    summary["spectral_l1_distance_vs_128"] = [
        0.5 * np.abs(spectra[g]["amplitude"].to_numpy(float) / spectra[g]["amplitude"].sum() - reference).sum()
        for g in summary["max_grid_size"]
    ]
    summary.to_csv(OUT / "mesh_sensitivity_summary.csv", index=False)

    combined = pd.DataFrame({"t2_ms": spectra[128]["t2_ms"]})
    for grid, spectrum in spectra.items():
        combined[f"normalized_amplitude_grid_{grid}"] = spectrum["normalized_amplitude"]
    combined.to_csv(OUT / "mesh_sensitivity_t2_spectra.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    colors = ["#6b7280", "#56a3a6", "#2f7d68", "#d9a441", "#c65d3b", "#7b5aa6"]
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0])
    ax_spectrum = fig.add_subplot(grid_spec[:, 0])
    ax_peak = fig.add_subplot(grid_spec[0, 1])
    ax_error = fig.add_subplot(grid_spec[1, 1])

    for color, (density, spectrum) in zip(colors, spectra.items()):
        ax_spectrum.plot(
            spectrum["t2_ms"], spectrum["normalized_amplitude"],
            color=color, lw=1.7, label=f"{density} x {density}",
        )
    ax_spectrum.set_xscale("log")
    ax_spectrum.set_xlim(100, 2000)
    ax_spectrum.set_xlabel(r"$T_2$ (ms)")
    ax_spectrum.set_ylabel("Normalized amplitude")
    ax_spectrum.set_title("a  T2 spectra across mesh densities", loc="left", fontweight="bold")
    ax_spectrum.legend(title="Simulation grid", ncol=2, loc="upper left")

    ax_peak.plot(summary["mesh_triangles"], summary["peak_t2_ms"], "o-", color="#245c73", lw=1.5)
    for _, row in summary.iterrows():
        ax_peak.annotate(str(int(row["max_grid_size"])), (row["mesh_triangles"], row["peak_t2_ms"]),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax_peak.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    ax_peak.set_xlabel("Mesh triangles")
    ax_peak.set_ylabel(r"Peak $T_2$ (ms)")
    ax_peak.set_title("b  Peak convergence", loc="left", fontweight="bold")

    ax_error.plot(summary["mesh_triangles"], summary["spectral_l1_distance_vs_128"], "o-", color="#b34b3f", lw=1.5)
    for _, row in summary.iterrows():
        ax_error.annotate(str(int(row["max_grid_size"])),
                          (row["mesh_triangles"], row["spectral_l1_distance_vs_128"]),
                          xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax_error.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    ax_error.set_xlabel("Mesh triangles")
    ax_error.set_ylabel("Spectral distance vs 128")
    ax_error.set_title("c  Full-spectrum convergence", loc="left", fontweight="bold")

    fig.suptitle(r"Berea slice 90 mesh-density sensitivity ($\alpha=0.1$)", fontsize=11, fontweight="bold")
    fig.savefig(OUT / "mesh_density_t2_sensitivity.png", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
