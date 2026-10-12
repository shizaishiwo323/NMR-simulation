"""Fit an effective rho2 and plot Sample 16 T2 spectra on one diameter axis."""

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


DEFAULT_RUN_DIR = Path("simulation_outputs/sample_16_porosity_grouped_nmr_topaxis_20260914")
DEFAULT_NETWORK_DIR = Path(
    r"E:\python\jupyter\codex\sip模拟\results\sample16_gpu_20260912\pore_network\network_parsed"
)


def t2_to_diameter_um(t2_ms: np.ndarray | float, rho2_um_per_s: float) -> np.ndarray:
    """Spherical fast-diffusion mapping: d(um) = 0.006*rho2(um/s)*T2(ms)."""
    return 0.006 * float(rho2_um_per_s) * np.asarray(t2_ms, dtype=float)


def fit_rho2(
    t2_ms: np.ndarray,
    amplitude: np.ndarray,
    pore_diameter_um: np.ndarray,
    pore_volume: np.ndarray,
    pore_peak_um: float,
    rho_grid: np.ndarray,
) -> pd.DataFrame:
    """Score rho2 by full-distribution distance plus main-peak mismatch in log space."""
    t2_peak_ms = float(t2_ms[int(np.argmax(amplitude))])
    pore_log = np.log10(pore_diameter_um)
    rows = []
    for rho2 in rho_grid:
        converted = t2_to_diameter_um(t2_ms, rho2)
        distribution_distance = float(
            wasserstein_distance(
                pore_log,
                np.log10(converted),
                u_weights=pore_volume,
                v_weights=amplitude,
            )
        )
        converted_peak_um = float(t2_to_diameter_um(t2_peak_ms, rho2))
        peak_error = abs(float(np.log10(converted_peak_um / pore_peak_um)))
        rows.append(
            {
                "rho2_um_per_s": float(rho2),
                "converted_nmr_main_peak_um": converted_peak_um,
                "distribution_wasserstein_log10_decades": distribution_distance,
                "main_peak_log10_error": peak_error,
                "combined_score": distribution_distance + peak_error,
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
        default=Path("simulation_outputs/sample_16_t2_equivalent_diameter_fit_20260914"),
    )
    parser.add_argument("--candidate-rhos", default="3.16,4.21,10")
    parser.add_argument("--fit-rho-min", type=float, default=0.5)
    parser.add_argument("--fit-rho-max", type=float, default=60.0)
    parser.add_argument("--fit-rho-step", type=float, default=0.05)
    parser.add_argument("--histogram-bins", type=int, default=80)
    return parser.parse_args()


def positive_arrays(values: pd.Series, weights: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    value_array = values.to_numpy(float)
    weight_array = weights.to_numpy(float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (value_array > 0) & (weight_array > 0)
    return value_array[valid], weight_array[valid]


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
    pore_diameter, pore_volume = positive_arrays(2e6 * pores["pore_radius_m"], pores["pore_volume_m3"])
    throat_diameter, throat_volume = positive_arrays(2e6 * throats["throat_radius_m"], throats["throat_volume_m3"])

    diameter_min = min(float(pore_diameter.min()), float(throat_diameter.min()))
    diameter_max = max(float(pore_diameter.max()), float(throat_diameter.max()))
    edges = np.geomspace(0.8 * diameter_min, 1.2 * diameter_max, args.histogram_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_hist, _ = np.histogram(pore_diameter, bins=edges, weights=pore_volume)
    throat_hist, _ = np.histogram(throat_diameter, bins=edges, weights=throat_volume)
    pore_hist = pore_hist / pore_hist.max()
    throat_hist = throat_hist / throat_hist.max()
    pore_peak_um = float(centers[int(np.argmax(pore_hist))])
    throat_peak_um = float(centers[int(np.argmax(throat_hist))])

    rho_count = int(round((args.fit_rho_max - args.fit_rho_min) / args.fit_rho_step)) + 1
    rho_grid = np.linspace(args.fit_rho_min, args.fit_rho_max, rho_count)
    scan = fit_rho2(exp_t2, exp_amp, pore_diameter, pore_volume, pore_peak_um, rho_grid)
    best_row = scan.loc[scan["combined_score"].idxmin()]
    distribution_best_row = scan.loc[scan["distribution_wasserstein_log10_decades"].idxmin()]
    common_range = scan[scan["rho2_um_per_s"].between(1.0 - 1e-12, 10.0 + 1e-12)]
    common_best_row = common_range.loc[common_range["combined_score"].idxmin()]
    best_rho2 = float(best_row["rho2_um_per_s"])

    candidate_rhos = [float(value) for value in args.candidate_rhos.split(",") if value.strip()]
    candidate_rhos = list(dict.fromkeys([*candidate_rhos, best_rho2]))
    candidate_comparison = fit_rho2(
        exp_t2,
        exp_amp,
        pore_diameter,
        pore_volume,
        pore_peak_um,
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
                "equivalent_diameter_um": float(diameter),
                "normalized_amplitude": float(amplitude),
            }
            for t2, diameter, amplitude in zip(exp_t2, t2_to_diameter_um(exp_t2, rho2), exp_norm)
        )
    converted_rows.extend(
        {
            "series": "porosity-grouped 2D NMR approximation",
            "rho2_um_per_s": simulation_rho2,
            "t2_ms": float(t2),
            "equivalent_diameter_um": float(diameter),
            "normalized_amplitude": float(amplitude),
        }
        for t2, diameter, amplitude in zip(sim_t2, t2_to_diameter_um(sim_t2, simulation_rho2), sim_norm)
    )

    scan_path = args.output_dir / "rho2_fit_scan.csv"
    candidate_path = args.output_dir / "rho2_candidate_comparison.csv"
    converted_path = args.output_dir / "t2_spectra_converted_to_equivalent_diameter.csv"
    histogram_path = args.output_dir / "pore_throat_diameter_histograms.csv"
    figure_path = args.output_dir / "sample16_t2_to_equivalent_diameter_single_axis.png"
    manifest_path = args.output_dir / "run_manifest.json"
    scan.to_csv(scan_path, index=False)
    candidate_comparison.to_csv(candidate_path, index=False)
    pd.DataFrame(converted_rows).to_csv(converted_path, index=False)
    pd.DataFrame(
        {
            "diameter_left_um": edges[:-1],
            "diameter_right_um": edges[1:],
            "diameter_center_um": centers,
            "pore_normalized_volume_weighted_frequency": pore_hist,
            "throat_normalized_volume_weighted_frequency": throat_hist,
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
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.stairs(
        pore_hist,
        edges,
        color="#58739d",
        linewidth=0,
        fill=True,
        alpha=0.13,
    )
    ax.stairs(
        throat_hist,
        edges,
        color="#c45d16",
        linewidth=0,
        fill=True,
        alpha=0.10,
    )

    comparison_colors = ["#b7d7d0", "#88beb5", "#4f9d92"]
    comparison_rhos = [rho for rho in candidate_rhos if not np.isclose(rho, best_rho2)]
    for color, rho2 in zip(comparison_colors, comparison_rhos):
        ax.plot(
            t2_to_diameter_um(exp_t2, rho2),
            exp_norm,
            color=color,
            linewidth=1.35,
            alpha=0.85,
            label=rf"experimental $T_2$: $\rho_2={rho2:g}$ $\mu$m/s",
        )
    ax.plot(
        t2_to_diameter_um(exp_t2, best_rho2),
        exp_norm,
        color="#00796b",
        linewidth=2.7,
        label=rf"experimental $T_2$: best effective $\rho_2={best_rho2:.2f}$ $\mu$m/s",
    )
    ax.plot(
        t2_to_diameter_um(sim_t2, simulation_rho2),
        sim_norm,
        color="#202936",
        linewidth=2.2,
        label=rf"grouped 2D NMR: input $\rho_2={simulation_rho2:g}$ $\mu$m/s",
    )
    ax.stairs(
        pore_hist,
        edges,
        color="#58739d",
        linewidth=1.8,
        label=rf"pore diameter (volume weighted; peak {pore_peak_um:.0f} $\mu$m)",
    )
    ax.stairs(
        throat_hist,
        edges,
        color="#c45d16",
        linewidth=1.7,
        linestyle="--",
        label=rf"throat diameter (volume weighted; peak {throat_peak_um:.0f} $\mu$m)",
    )

    converted_peak_um = float(best_row["converted_nmr_main_peak_um"])
    ax.axvline(pore_peak_um, color="#58739d", linewidth=1.1, linestyle=":", alpha=0.9)
    ax.axvline(throat_peak_um, color="#c45d16", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.scatter([converted_peak_um], [1.0], color="#00796b", s=28, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 3000)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"Equivalent diameter, $d$ ($\mu$m)")
    ax.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax.set_title(r"Sample 16: $T_2$-derived equivalent diameter vs pore-network sizes")
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
        rf"sphere model: $d=0.006\rho_2T_2$" "\n"
        rf"best peak: {converted_peak_um:.0f} $\mu$m; pore peak: {pore_peak_um:.0f} $\mu$m" "\n"
        rf"common 1–10 $\mu$m/s range: best at {float(common_best_row['rho2_um_per_s']):.0f} $\mu$m/s boundary",
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

    manifest = {
        "method": "Spherical fast-diffusion mapping d_um=0.006*rho2_um_per_s*T2_ms. Effective rho2 was scanned against the volume-weighted pore-body distribution.",
        "fit_score": "Wasserstein distance between log10 pore and converted-NMR distributions plus absolute log10 main-peak mismatch; both terms are in decades.",
        "fit_target": "pnextract pore-body diameter weighted by pore_volume_m3; throat diameter is shown but not fitted.",
        "candidate_rho2_um_per_s_shown": candidate_rhos,
        "candidate_fit_metrics": candidate_comparison.to_dict(orient="records"),
        "scan_range_rho2_um_per_s": [args.fit_rho_min, args.fit_rho_max, args.fit_rho_step],
        "best_effective_rho2_um_per_s": best_rho2,
        "best_combined_score": float(best_row["combined_score"]),
        "full_distribution_only_best_rho2_um_per_s": float(distribution_best_row["rho2_um_per_s"]),
        "common_carbonate_range_um_per_s": [1.0, 10.0],
        "best_within_common_range_um_per_s": float(common_best_row["rho2_um_per_s"]),
        "simulation_input_rho2_um_per_s": simulation_rho2,
        "experimental_t2_main_peak_ms": float(exp_t2[int(np.argmax(exp_amp))]),
        "converted_experimental_main_peak_um": converted_peak_um,
        "pore_distribution_peak_um": pore_peak_um,
        "throat_distribution_peak_um": throat_peak_um,
        "assumptions": [
            "fast diffusion",
            "surface relaxation dominates",
            "spherical equivalent pore geometry with S/V=6/d",
            "bulk and internal-gradient relaxation are neglected in this conversion",
            "fitted rho2 is an effective structure-matching parameter, not a directly measured intrinsic surface relaxivity",
        ],
        "model_dimension_note": "The simulated spectrum is from a porosity-grouped 2D representative-slice approximation to the 3D stack, not a full 3D NMR solve.",
        "inputs": {
            "experiment_spectrum": str(experiment_path.resolve()),
            "simulation_spectrum": str(simulation_path.resolve()),
            "pores": str(pores_path.resolve()),
            "throats": str(throats_path.resolve()),
            "source_manifest": str(source_manifest_path.resolve()),
        },
        "outputs": {
            "figure_png": str(figure_path.resolve()),
            "rho2_fit_scan_csv": str(scan_path.resolve()),
            "rho2_candidate_comparison_csv": str(candidate_path.resolve()),
            "converted_spectra_csv": str(converted_path.resolve()),
            "histograms_csv": str(histogram_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    assert np.isclose(t2_to_diameter_um(1000.0, 3.16), 18.96)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
