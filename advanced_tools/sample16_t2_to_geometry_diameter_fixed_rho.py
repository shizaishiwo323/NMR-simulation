"""Plot Sample 16 T2 spectra and pore-network sizes on one diameter axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.sample16_t2_to_equivalent_diameter_fit import (
    DEFAULT_NETWORK_DIR,
    DEFAULT_RUN_DIR,
    positive_arrays,
)


def t2_to_radius_um(
    t2_ms: np.ndarray | float,
    rho2_um_per_s: float,
    geometry_factor: float,
) -> np.ndarray:
    """Return r=geometry_factor*rho2*T2 with T2 supplied in milliseconds."""
    return 0.001 * geometry_factor * float(rho2_um_per_s) * np.asarray(t2_ms, dtype=float)


def t2_to_diameter_um(
    t2_ms: np.ndarray | float,
    rho2_um_per_s: float,
    geometry_factor: float,
) -> np.ndarray:
    return 2.0 * t2_to_radius_um(t2_ms, rho2_um_per_s, geometry_factor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_outputs/sample_16_geometry_diameter_rho2_4p21_20260914"),
    )
    parser.add_argument("--rho2", type=float, default=4.21)
    parser.add_argument("--histogram-bins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment_path = args.run_dir / "experiment_t2_inversion_fixed_alpha_476p4.csv"
    simulation_path = args.run_dir / "porosity_group_weighted_t2_inversion_fixed_alpha_476p4.csv"
    source_manifest_path = args.run_dir / "run_manifest.json"
    pores_path = args.network_dir / "pores.csv"
    throats_path = args.network_dir / "throats.csv"

    experiment = pd.read_csv(experiment_path)
    simulation = pd.read_csv(simulation_path)
    pores = pd.read_csv(pores_path)
    throats = pd.read_csv(throats_path)

    exp_t2, exp_amp = positive_arrays(experiment["t2_ms"], experiment["amplitude"])
    sim_t2, sim_amp = positive_arrays(simulation["t2_ms"], simulation["amplitude"])
    pore_diameter, pore_volume = positive_arrays(
        2e6 * pores["pore_radius_m"], pores["pore_volume_m3"]
    )
    throat_diameter, throat_volume = positive_arrays(
        2e6 * throats["throat_radius_m"], throats["throat_volume_m3"]
    )
    exp_norm = exp_amp / exp_amp.max()
    sim_norm = sim_amp / sim_amp.max()

    edges = np.geomspace(
        0.8 * min(float(pore_diameter.min()), float(throat_diameter.min())),
        1.2 * max(float(pore_diameter.max()), float(throat_diameter.max())),
        args.histogram_bins + 1,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_hist_raw, _ = np.histogram(pore_diameter, bins=edges, weights=pore_volume)
    throat_hist_raw, _ = np.histogram(throat_diameter, bins=edges, weights=throat_volume)
    pore_hist = pore_hist_raw / pore_hist_raw.max()
    throat_hist = throat_hist_raw / throat_hist_raw.max()
    pore_peak = float(centers[int(np.argmax(pore_hist_raw))])
    throat_peak = float(centers[int(np.argmax(throat_hist_raw))])

    sphere_factor = 3.0
    cylinder_factor = 2.0
    exp_peak_t2 = float(exp_t2[int(np.argmax(exp_amp))])
    sim_peak_t2 = float(sim_t2[int(np.argmax(sim_amp))])
    exp_sphere_peak = float(t2_to_diameter_um(exp_peak_t2, args.rho2, sphere_factor))
    exp_cylinder_peak = float(t2_to_diameter_um(exp_peak_t2, args.rho2, cylinder_factor))
    sim_sphere_peak = float(t2_to_diameter_um(sim_peak_t2, args.rho2, sphere_factor))
    sim_cylinder_peak = float(t2_to_diameter_um(sim_peak_t2, args.rho2, cylinder_factor))

    converted_path = args.output_dir / "t2_spectra_geometry_diameters.csv"
    histogram_path = args.output_dir / "pore_throat_diameter_histograms.csv"
    peak_path = args.output_dir / "peak_position_summary.csv"
    figure_path = args.output_dir / "sample16_t2_geometry_diameter_rho2_4p21.png"
    manifest_path = args.output_dir / "run_manifest.json"

    converted_frames = []
    for label, t2, amplitude in (
        ("experimental T2", exp_t2, exp_norm),
        ("porosity-grouped 2D NMR approximation", sim_t2, sim_norm),
    ):
        converted_frames.append(
            pd.DataFrame(
                {
                    "series": label,
                    "t2_ms": t2,
                    "normalized_amplitude": amplitude,
                    "sphere_radius_um_alpha3": t2_to_radius_um(t2, args.rho2, sphere_factor),
                    "sphere_diameter_um_alpha3": t2_to_diameter_um(t2, args.rho2, sphere_factor),
                    "cylinder_radius_um_alpha2": t2_to_radius_um(t2, args.rho2, cylinder_factor),
                    "cylinder_diameter_um_alpha2": t2_to_diameter_um(
                        t2, args.rho2, cylinder_factor
                    ),
                }
            )
        )
    pd.concat(converted_frames, ignore_index=True).to_csv(converted_path, index=False)
    pd.DataFrame(
        {
            "diameter_left_um": edges[:-1],
            "diameter_right_um": edges[1:],
            "diameter_center_um": centers,
            "pore_volume_weighted_frequency": pore_hist_raw,
            "pore_normalized_volume_weighted_frequency": pore_hist,
            "throat_volume_weighted_frequency": throat_hist_raw,
            "throat_normalized_volume_weighted_frequency": throat_hist,
        }
    ).to_csv(histogram_path, index=False)
    pd.DataFrame(
        [
            {"series": "pore-network pore body", "peak_diameter_um": pore_peak},
            {"series": "pore-network throat", "peak_diameter_um": throat_peak},
            {
                "series": "experimental T2, sphere alpha=3",
                "t2_peak_ms": exp_peak_t2,
                "peak_diameter_um": exp_sphere_peak,
            },
            {
                "series": "experimental T2, cylinder alpha=2",
                "t2_peak_ms": exp_peak_t2,
                "peak_diameter_um": exp_cylinder_peak,
            },
            {
                "series": "grouped 2D NMR, sphere alpha=3",
                "t2_peak_ms": sim_peak_t2,
                "peak_diameter_um": sim_sphere_peak,
            },
            {
                "series": "grouped 2D NMR, cylinder alpha=2",
                "t2_peak_ms": sim_peak_t2,
                "peak_diameter_um": sim_cylinder_peak,
            },
        ]
    ).to_csv(peak_path, index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "legend.frameon": True,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    pore_color = "#4f6f9f"
    throat_color = "#c45d16"
    sim_color = "#273444"

    ax.stairs(pore_hist, edges, color=pore_color, fill=True, alpha=0.13, linewidth=0)
    ax.stairs(throat_hist, edges, color=throat_color, fill=True, alpha=0.10, linewidth=0)
    ax.stairs(
        pore_hist,
        edges,
        color=pore_color,
        linewidth=1.8,
        label=rf"PNM pore-body diameter (peak {pore_peak:.0f} $\mu$m)",
    )
    ax.stairs(
        throat_hist,
        edges,
        color=throat_color,
        linewidth=1.7,
        linestyle="--",
        label=rf"PNM throat diameter (peak {throat_peak:.0f} $\mu$m)",
    )
    ax.plot(
        t2_to_diameter_um(exp_t2, args.rho2, sphere_factor),
        exp_norm,
        color="#00796b",
        linewidth=2.5,
        label=rf"experimental $T_2 \rightarrow$ sphere $d_p$ (peak {exp_sphere_peak:.1f} $\mu$m)",
    )
    ax.plot(
        t2_to_diameter_um(exp_t2, args.rho2, cylinder_factor),
        exp_norm,
        color="#7a4f9a",
        linewidth=2.2,
        linestyle="--",
        label=rf"experimental $T_2 \rightarrow$ cylinder $d_t$ (peak {exp_cylinder_peak:.1f} $\mu$m)",
    )
    ax.plot(
        t2_to_diameter_um(sim_t2, args.rho2, sphere_factor),
        sim_norm,
        color=sim_color,
        linewidth=1.8,
        alpha=0.82,
        label=rf"grouped 2D NMR $\rightarrow$ sphere $d_p$ (peak {sim_sphere_peak:.1f} $\mu$m)",
    )
    ax.plot(
        t2_to_diameter_um(sim_t2, args.rho2, cylinder_factor),
        sim_norm,
        color=sim_color,
        linewidth=1.7,
        linestyle=":",
        alpha=0.82,
        label=rf"grouped 2D NMR $\rightarrow$ cylinder $d_t$ (peak {sim_cylinder_peak:.1f} $\mu$m)",
    )

    ax.axvline(pore_peak, color=pore_color, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.axvline(throat_peak, color=throat_color, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.scatter([exp_sphere_peak], [1.0], color="#00796b", s=28, zorder=5)
    ax.scatter([exp_cylinder_peak], [1.0], color="#7a4f9a", s=28, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 3000)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"Diameter, $d$ ($\mu$m)")
    ax.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax.set_title(r"Sample 16: geometry-aware $T_2$-to-diameter mapping at $\rho_2=4.21$ $\mu$m/s")
    ax.grid(which="major", alpha=0.20)
    ax.grid(which="minor", axis="x", alpha=0.08)
    ax.legend(
        loc="upper left",
        fontsize=7.5,
        ncol=2,
        columnspacing=1.0,
        handlelength=3.0,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#cccccc",
    )
    ax.text(
        0.985,
        0.045,
        r"sphere: $r_p=3\rho_2T_2$, $d_p=6\rho_2T_2$" "\n"
        r"cylinder: $r_t=2\rho_2T_2$, $d_t=4\rho_2T_2$" "\n"
        rf"experimental peak: {exp_peak_t2:.0f} ms $\rightarrow$ {exp_sphere_peak:.1f} / {exp_cylinder_peak:.1f} $\mu$m",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 5},
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "method": "Fixed-rho2 fast-diffusion conversion on one physical diameter axis.",
        "rho2_um_per_s": args.rho2,
        "geometry": {
            "pore_body": "sphere: r=3*rho2*T2 and d=6*rho2*T2",
            "throat": "open cylinder: r=2*rho2*T2 and d=4*rho2*T2",
        },
        "diameter_conversion_um_per_ms": {
            "sphere": 0.006 * args.rho2,
            "cylinder": 0.004 * args.rho2,
        },
        "peak_metrics": {
            "experimental_t2_ms": exp_peak_t2,
            "grouped_2d_nmr_t2_ms": sim_peak_t2,
            "experimental_sphere_diameter_um": exp_sphere_peak,
            "experimental_cylinder_diameter_um": exp_cylinder_peak,
            "grouped_2d_nmr_sphere_diameter_um": sim_sphere_peak,
            "grouped_2d_nmr_cylinder_diameter_um": sim_cylinder_peak,
            "pore_network_pore_diameter_um": pore_peak,
            "pore_network_throat_diameter_um": throat_peak,
            "pore_peak_ratio_pnm_to_experimental_nmr": pore_peak / exp_sphere_peak,
            "throat_peak_ratio_pnm_to_experimental_nmr": throat_peak / exp_cylinder_peak,
        },
        "assumptions": [
            "fast diffusion",
            "surface relaxation dominates",
            "rho2=4.21 um/s is shared by pore-body and throat water-rock surfaces",
            "pore bodies are represented as spheres and throats as open circular cylinders",
            "bulk and internal-gradient relaxation are neglected in this post-processing conversion",
        ],
        "model_dimension_note": "The simulated spectrum is a porosity-grouped 2D representative-slice approximation to the 3D stack, not a full 3D NMR solve.",
        "inputs": {
            "experiment_spectrum": str(experiment_path.resolve()),
            "simulation_spectrum": str(simulation_path.resolve()),
            "pores": str(pores_path.resolve()),
            "throats": str(throats_path.resolve()),
            "source_manifest": str(source_manifest_path.resolve()),
        },
        "outputs": {
            "figure_png": str(figure_path.resolve()),
            "converted_spectra_csv": str(converted_path.resolve()),
            "histograms_csv": str(histogram_path.resolve()),
            "peak_summary_csv": str(peak_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    assert np.isclose(t2_to_radius_um(1000.0, 4.21, 3.0), 12.63)
    assert np.isclose(t2_to_radius_um(1000.0, 4.21, 2.0), 8.42)
    assert np.isclose(t2_to_diameter_um(1000.0, 4.21, 3.0), 25.26)
    assert np.isclose(t2_to_diameter_um(1000.0, 4.21, 2.0), 16.84)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
