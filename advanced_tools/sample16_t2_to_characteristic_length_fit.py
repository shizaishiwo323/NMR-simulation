"""Fit one rho2 for Sample 16 on a pore/throat characteristic-length axis."""

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
from scipy.stats import wasserstein_distance

from advanced_tools.sample16_t2_to_equivalent_diameter_fit import (
    DEFAULT_NETWORK_DIR,
    DEFAULT_RUN_DIR,
    positive_arrays,
)


def t2_to_characteristic_length_um(t2_ms: np.ndarray | float, rho2_um_per_s: float) -> np.ndarray:
    """Fast-diffusion mapping Lc=V/S=rho2*T2 with um, s, and ms units."""
    return 0.001 * float(rho2_um_per_s) * np.asarray(t2_ms, dtype=float)


def sphere_diameter_to_lc_um(diameter_um: np.ndarray | float) -> np.ndarray:
    return np.asarray(diameter_um, dtype=float) / 6.0


def open_cylinder_diameter_to_lc_um(diameter_um: np.ndarray | float) -> np.ndarray:
    return np.asarray(diameter_um, dtype=float) / 4.0


def score_component(
    t2_ms: np.ndarray,
    amplitude: np.ndarray,
    target_lc_um: np.ndarray,
    target_volume: np.ndarray,
    target_peak_um: float,
    rho2_um_per_s: float,
) -> tuple[float, float, float]:
    converted = t2_to_characteristic_length_um(t2_ms, rho2_um_per_s)
    distribution_distance = float(
        wasserstein_distance(
            np.log10(target_lc_um),
            np.log10(converted),
            u_weights=target_volume,
            v_weights=amplitude,
        )
    )
    converted_peak = float(converted[int(np.argmax(amplitude))])
    peak_error = abs(float(np.log10(converted_peak / target_peak_um)))
    return distribution_distance, peak_error, distribution_distance + peak_error


def fit_shared_rho2(
    t2_ms: np.ndarray,
    amplitude: np.ndarray,
    pore_lc_um: np.ndarray,
    pore_volume: np.ndarray,
    pore_peak_um: float,
    throat_lc_um: np.ndarray,
    throat_volume: np.ndarray,
    throat_peak_um: float,
    rho_grid: np.ndarray,
) -> pd.DataFrame:
    pore_fraction = float(pore_volume.sum() / (pore_volume.sum() + throat_volume.sum()))
    rows = []
    for rho2 in rho_grid:
        pore_w1, pore_peak_error, pore_score = score_component(
            t2_ms, amplitude, pore_lc_um, pore_volume, pore_peak_um, rho2
        )
        throat_w1, throat_peak_error, throat_score = score_component(
            t2_ms, amplitude, throat_lc_um, throat_volume, throat_peak_um, rho2
        )
        rows.append(
            {
                "rho2_um_per_s": float(rho2),
                "converted_nmr_main_peak_lc_um": float(
                    t2_to_characteristic_length_um(t2_ms[int(np.argmax(amplitude))], rho2)
                ),
                "pore_distribution_wasserstein_log10_decades": pore_w1,
                "pore_main_peak_log10_error": pore_peak_error,
                "pore_score": pore_score,
                "throat_distribution_wasserstein_log10_decades": throat_w1,
                "throat_main_peak_log10_error": throat_peak_error,
                "throat_score": throat_score,
                "pore_volume_fraction": pore_fraction,
                "throat_volume_fraction": 1.0 - pore_fraction,
                "combined_volume_weighted_score": pore_fraction * pore_score
                + (1.0 - pore_fraction) * throat_score,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_outputs/sample_16_characteristic_length_fit_20260914"),
    )
    parser.add_argument("--candidate-rhos", default="3.16,4.21,10")
    parser.add_argument("--fit-rho-min", type=float, default=0.5)
    parser.add_argument("--fit-rho-max", type=float, default=60.0)
    parser.add_argument("--fit-rho-step", type=float, default=0.05)
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
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    simulation_rho2 = 1000.0 * float(source_manifest["simulation_params"]["rho_solid_um_per_ms"])

    exp_t2, exp_amp = positive_arrays(experiment["t2_ms"], experiment["amplitude"])
    sim_t2, sim_amp = positive_arrays(simulation["t2_ms"], simulation["amplitude"])
    pore_radius_um, pore_volume = positive_arrays(1e6 * pores["pore_radius_m"], pores["pore_volume_m3"])
    throat_radius_um, throat_volume = positive_arrays(
        1e6 * throats["throat_radius_m"], throats["throat_volume_m3"]
    )
    pore_lc_um = sphere_diameter_to_lc_um(2.0 * pore_radius_um)
    throat_lc_um = open_cylinder_diameter_to_lc_um(2.0 * throat_radius_um)

    edges = np.geomspace(
        0.8 * min(float(pore_lc_um.min()), float(throat_lc_um.min())),
        1.2 * max(float(pore_lc_um.max()), float(throat_lc_um.max())),
        args.histogram_bins + 1,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_hist, _ = np.histogram(pore_lc_um, bins=edges, weights=pore_volume)
    throat_hist, _ = np.histogram(throat_lc_um, bins=edges, weights=throat_volume)
    pore_hist_normalized = pore_hist / pore_hist.max()
    throat_hist_normalized = throat_hist / throat_hist.max()
    pore_peak_um = float(centers[int(np.argmax(pore_hist))])
    throat_peak_um = float(centers[int(np.argmax(throat_hist))])

    rho_count = int(round((args.fit_rho_max - args.fit_rho_min) / args.fit_rho_step)) + 1
    rho_grid = np.linspace(args.fit_rho_min, args.fit_rho_max, rho_count)
    scan = fit_shared_rho2(
        exp_t2,
        exp_amp,
        pore_lc_um,
        pore_volume,
        pore_peak_um,
        throat_lc_um,
        throat_volume,
        throat_peak_um,
        rho_grid,
    )
    best_row = scan.loc[scan["combined_volume_weighted_score"].idxmin()]
    pore_best_row = scan.loc[scan["pore_score"].idxmin()]
    throat_best_row = scan.loc[scan["throat_score"].idxmin()]
    common_scan = scan[scan["rho2_um_per_s"].between(1.0 - 1e-12, 10.0 + 1e-12)]
    common_best_row = common_scan.loc[common_scan["combined_volume_weighted_score"].idxmin()]
    best_rho2 = float(best_row["rho2_um_per_s"])

    candidate_rhos = [float(value) for value in args.candidate_rhos.split(",") if value.strip()]
    candidate_rhos = list(dict.fromkeys([*candidate_rhos, best_rho2]))
    candidate_comparison = fit_shared_rho2(
        exp_t2,
        exp_amp,
        pore_lc_um,
        pore_volume,
        pore_peak_um,
        throat_lc_um,
        throat_volume,
        throat_peak_um,
        np.asarray(candidate_rhos),
    )
    exp_norm = exp_amp / exp_amp.max()
    sim_norm = sim_amp / sim_amp.max()

    converted_rows = []
    for rho2 in candidate_rhos:
        converted_rows.extend(
            {
                "series": "experimental T2",
                "rho2_um_per_s": rho2,
                "t2_ms": float(t2),
                "characteristic_length_um": float(lc),
                "normalized_amplitude": float(amplitude),
            }
            for t2, lc, amplitude in zip(
                exp_t2, t2_to_characteristic_length_um(exp_t2, rho2), exp_norm
            )
        )
    converted_rows.extend(
        {
            "series": "porosity-grouped 2D NMR approximation",
            "rho2_um_per_s": simulation_rho2,
            "t2_ms": float(t2),
            "characteristic_length_um": float(lc),
            "normalized_amplitude": float(amplitude),
        }
        for t2, lc, amplitude in zip(
            sim_t2, t2_to_characteristic_length_um(sim_t2, simulation_rho2), sim_norm
        )
    )

    scan_path = args.output_dir / "rho2_shared_fit_scan.csv"
    candidate_path = args.output_dir / "rho2_candidate_comparison.csv"
    histogram_path = args.output_dir / "pore_throat_characteristic_length_histograms.csv"
    converted_path = args.output_dir / "t2_spectra_converted_to_characteristic_length.csv"
    figure_path = args.output_dir / "sample16_t2_to_characteristic_length_single_axis.png"
    manifest_path = args.output_dir / "run_manifest.json"
    scan.to_csv(scan_path, index=False)
    candidate_comparison.to_csv(candidate_path, index=False)
    pd.DataFrame(converted_rows).to_csv(converted_path, index=False)
    pd.DataFrame(
        {
            "lc_left_um": edges[:-1],
            "lc_right_um": edges[1:],
            "lc_center_um": centers,
            "pore_volume_weighted_frequency": pore_hist,
            "pore_normalized_volume_weighted_frequency": pore_hist_normalized,
            "throat_volume_weighted_frequency": throat_hist,
            "throat_normalized_volume_weighted_frequency": throat_hist_normalized,
        }
    ).to_csv(histogram_path, index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.stairs(pore_hist_normalized, edges, color="#58739d", linewidth=0, fill=True, alpha=0.13)
    ax.stairs(throat_hist_normalized, edges, color="#c45d16", linewidth=0, fill=True, alpha=0.10)

    comparison_colors = ["#b7d7d0", "#88beb5", "#4f9d92"]
    comparison_rhos = [rho for rho in candidate_rhos if not np.isclose(rho, best_rho2)]
    for color, rho2 in zip(comparison_colors, comparison_rhos):
        ax.plot(
            t2_to_characteristic_length_um(exp_t2, rho2),
            exp_norm,
            color=color,
            linewidth=1.35,
            alpha=0.85,
            label=rf"experimental $T_2$: $\rho_2={rho2:g}$ $\mu$m/s",
        )
    ax.plot(
        t2_to_characteristic_length_um(exp_t2, best_rho2),
        exp_norm,
        color="#00796b",
        linewidth=2.7,
        label=rf"experimental $T_2$: shared best $\rho_2={best_rho2:.2f}$ $\mu$m/s",
    )
    ax.plot(
        t2_to_characteristic_length_um(sim_t2, simulation_rho2),
        sim_norm,
        color="#202936",
        linewidth=2.2,
        label=rf"grouped 2D NMR: input $\rho_2={simulation_rho2:g}$ $\mu$m/s",
    )
    ax.stairs(
        pore_hist_normalized,
        edges,
        color="#58739d",
        linewidth=1.8,
        label=rf"pore body: $L_c=d_p/6$ (peak {pore_peak_um:.1f} $\mu$m)",
    )
    ax.stairs(
        throat_hist_normalized,
        edges,
        color="#c45d16",
        linewidth=1.7,
        linestyle="--",
        label=rf"open throat: $L_c=d_t/4$ (peak {throat_peak_um:.1f} $\mu$m)",
    )

    converted_peak_um = float(best_row["converted_nmr_main_peak_lc_um"])
    ax.axvline(pore_peak_um, color="#58739d", linewidth=1.0, linestyle=":", alpha=0.9)
    ax.axvline(throat_peak_um, color="#c45d16", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.scatter([converted_peak_um], [1.0], color="#00796b", s=28, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 600)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"Characteristic length, $L_c=V/S$ ($\mu$m)")
    ax.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax.set_title(r"Sample 16: shared-$\rho_2$ mapping on a pore/throat $V/S$ axis")
    ax.grid(which="major", alpha=0.20)
    ax.grid(which="minor", axis="x", alpha=0.09)
    handles, labels = ax.get_legend_handles_labels()
    order = [-2, -1, 0, 1, 2, 3, 4]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper left",
        fontsize=8.0,
        ncol=2,
        columnspacing=1.2,
        handlelength=3.0,
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        edgecolor="#cccccc",
    )
    ax.text(
        0.985,
        0.045,
        rf"shared best $\rho_2={best_rho2:.2f}$ $\mu$m/s" "\n"
        rf"NMR peak {converted_peak_um:.1f}; pore {pore_peak_um:.1f}; throat {throat_peak_um:.1f} $\mu$m" "\n"
        rf"pore score {float(best_row['pore_score']):.3f}; throat score {float(best_row['throat_score']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.90, "pad": 5},
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pore_fraction = float(best_row["pore_volume_fraction"])
    manifest = {
        "method": "One shared effective rho2 maps NMR T2 to Lc=V/S. Pore bodies use Lc=d/6; open cylindrical throats use Lc=d/4.",
        "fit_score": "Each component score is log10-Wasserstein distribution distance plus absolute log10 main-peak mismatch. The joint score is weighted by retained pore/throat network volume.",
        "best_shared_effective_rho2_um_per_s": best_rho2,
        "pore_only_best_rho2_um_per_s": float(pore_best_row["rho2_um_per_s"]),
        "throat_only_best_rho2_um_per_s": float(throat_best_row["rho2_um_per_s"]),
        "best_within_common_1_to_10_um_per_s": float(common_best_row["rho2_um_per_s"]),
        "simulation_input_rho2_um_per_s": simulation_rho2,
        "experimental_t2_main_peak_ms": float(exp_t2[int(np.argmax(exp_amp))]),
        "converted_experimental_main_peak_lc_um": converted_peak_um,
        "pore_lc_peak_um": pore_peak_um,
        "throat_lc_peak_um": throat_peak_um,
        "pore_score_at_shared_best": float(best_row["pore_score"]),
        "throat_score_at_shared_best": float(best_row["throat_score"]),
        "combined_volume_weighted_score": float(best_row["combined_volume_weighted_score"]),
        "network_volume_fractions": {"pore": pore_fraction, "throat": 1.0 - pore_fraction},
        "candidate_fit_metrics": candidate_comparison.to_dict(orient="records"),
        "geometry": {
            "pore_body": "sphere, Lc=V/S=R/3=d/6",
            "throat": "open circular cylinder using lateral water-rock surface only, Lc=V/S=R/2=d/4",
            "throat_end_caps": "excluded because throat ends connect to pore bodies and are not water-rock interfaces",
        },
        "assumptions": [
            "fast diffusion",
            "surface relaxation dominates",
            "one shared effective rho2 for pore and throat water-rock surfaces",
            "bulk and internal-gradient relaxation are neglected in this post-processing conversion",
            "the fitted rho2 is a structure-matching effective parameter, not a directly measured intrinsic relaxivity",
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
            "rho2_shared_fit_scan_csv": str(scan_path.resolve()),
            "rho2_candidate_comparison_csv": str(candidate_path.resolve()),
            "histograms_csv": str(histogram_path.resolve()),
            "converted_spectra_csv": str(converted_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    assert np.isclose(t2_to_characteristic_length_um(1000.0, 3.16), 3.16)
    assert np.isclose(sphere_diameter_to_lc_um(6.0), 1.0)
    assert np.isclose(open_cylinder_diameter_to_lc_um(4.0), 1.0)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
