"""Calibrate Sample 16 rho2 by Zhang et al.'s cumulative-volume method."""

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
    positive_arrays,
)
from advanced_tools.sample16_t2_to_geometry_diameter_fixed_rho import (
    t2_to_diameter_um,
)
from advanced_tools.compare_experimental_t2_with_sim_average import load_experiment_decay
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_EXPERIMENT = Path(
    r"E:\python\jupyter\codex\sip模拟\data\ct_backed_samples_raw_copy_20260605"
    r"\sample_16_Grainstone\NMR_original\16-1\T2CPMG\Spectra\spectrum.csv"
)
DEFAULT_COMBINED = Path(
    r"E:\python\jupyter\codex\sip模拟\data\ct_backed_samples_raw_copy_20260605"
    r"\sample_16_Grainstone\NMR_combined\16\T2CPMG\Spectra\spectrum.csv"
)
DEFAULT_SIMULATION = Path(
    "simulation_outputs/sample_16_porosity_grouped_nmr_rho2_14p63_20260914/"
    "porosity_group_weighted_t2_inversion_fixed_alpha_476p4.csv"
)
DEFAULT_EXPERIMENT_DECAY = DEFAULT_EXPERIMENT.parents[1] / "data.csv"
DEFAULT_SIMULATION_DECAY = Path(
    "simulation_outputs/sample_16_porosity_grouped_nmr_rho2_14p63_20260914/"
    "group_weighted_normalized_decay.csv"
)


def weighted_log_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return float(10 ** np.interp(q, cumulative, np.log10(values)))


def calibration(
    pore_radius_um: np.ndarray,
    pore_volume: np.ndarray,
    t2_ms: np.ndarray,
    amplitude: np.ndarray,
    ct_porosity_pct: float,
    total_porosity_pct: float,
    alignment_vc: float,
) -> dict[str, float]:
    resolved_fraction = ct_porosity_pct / total_porosity_pct
    unresolved_fraction = 1.0 - resolved_fraction
    if not unresolved_fraction <= alignment_vc <= 1.0:
        raise ValueError(
            f"alignment_vc must be within [{unresolved_fraction:.6g}, 1]."
        )
    raw_ct_quantile = (alignment_vc - unresolved_fraction) / resolved_fraction
    radius_um = weighted_log_quantile(pore_radius_um, pore_volume, raw_ct_quantile)
    t2_ms_at_alignment = weighted_log_quantile(t2_ms, amplitude, alignment_vc)
    return {
        "ct_porosity_pct": ct_porosity_pct,
        "total_porosity_pct": total_porosity_pct,
        "ct_resolved_fraction": resolved_fraction,
        "ct_curve_start_vc": unresolved_fraction,
        "alignment_vc": alignment_vc,
        "raw_ct_quantile_at_alignment": raw_ct_quantile,
        "ct_radius_at_alignment_um": radius_um,
        "nmr_t2_at_alignment_ms": t2_ms_at_alignment,
        "rho2_sphere_alpha3_um_per_s": radius_um / (3.0 * t2_ms_at_alignment * 1e-3),
        "rho2_cylinder_alpha2_um_per_s": radius_um / (2.0 * t2_ms_at_alignment * 1e-3),
    }


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, header=None, names=["t2_ms", "amplitude"])
    t2_ms = frame["t2_ms"].to_numpy(float)
    amplitude = frame["amplitude"].to_numpy(float)
    valid = (
        np.isfinite(t2_ms)
        & np.isfinite(amplitude)
        & (t2_ms > 0)
        & (amplitude >= 0)
    )
    return t2_ms[valid], amplitude[valid]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--simulation-spectrum", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--experiment-decay", type=Path)
    parser.add_argument("--simulation-decay", type=Path)
    parser.add_argument("--experiment-alpha", type=float)
    parser.add_argument("--simulation-alpha", type=float)
    parser.add_argument("--simulation-forward-rho2-um-per-s", type=float)
    parser.add_argument("--rho2-override", type=float)
    parser.add_argument(
        "--figure-name",
        default="sample16_rho2_14p63_with_rerun_simulation.png",
    )
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--internal-throats-only", action="store_true")
    parser.add_argument("--ct-source-label", default="PNM")
    parser.add_argument("--ct-porosity", type=float, default=19.1)
    parser.add_argument("--total-porosity", type=float, default=23.88)
    parser.add_argument("--alignment-vc", type=float, default=0.5)
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument("--diameter-max-um", type=float, default=1500.0)
    parser.add_argument("--raw-y", action="store_true")
    parser.add_argument("--raw-ct-y", action="store_true")
    parser.add_argument("--reference-right-style", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_outputs/sample_16_rho2_cumulative_vc_calibration_20260914"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulation_forward_rho2_um_per_s is not None and args.simulation_forward_rho2_um_per_s <= 0:
        raise ValueError("simulation-forward-rho2-um-per-s must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vendor_exp_t2, vendor_exp_amp = read_spectrum(args.experiment)
    vendor_positive = vendor_exp_amp > 0
    vendor_t2 = vendor_exp_t2[vendor_positive]
    vendor_amp = vendor_exp_amp[vendor_positive]
    exp_plot_t2, exp_plot_amp = vendor_exp_t2, vendor_exp_amp
    if args.experiment_alpha is not None:
        if args.experiment_decay is None:
            raise ValueError("--experiment-decay is required with --experiment-alpha")
        exp_time, exp_signal, _ = load_experiment_decay(args.experiment_decay)
        exp_result = invert_single_signal_nnls(
            exp_time,
            exp_signal,
            signal_name="sample16_experiment",
            config=NnlsConfig(
                num_bins=200,
                regularization=args.experiment_alpha,
                t2_min_ms=0.01,
                t2_max_ms=100000.0,
            ),
        )
        exp_plot_t2 = exp_result.t2_bins_ms
        exp_plot_amp = exp_result.spectrum
        pd.DataFrame(
            {
                "t2_ms": exp_plot_t2,
                "amplitude": exp_plot_amp,
                "normalized_amplitude": exp_plot_amp / max(float(exp_plot_amp.max()), 1e-30),
            }
        ).to_csv(
            args.output_dir / f"experiment_t2_inversion_alpha_{args.experiment_alpha:g}.csv",
            index=False,
        )
    combined_plot_t2, combined_plot_amp = read_spectrum(args.combined)
    exp_positive = exp_plot_amp > 0
    combined_positive = combined_plot_amp > 0
    exp_t2, exp_amp = exp_plot_t2[exp_positive], exp_plot_amp[exp_positive]
    combined_t2 = combined_plot_t2[combined_positive]
    combined_amp = combined_plot_amp[combined_positive]
    if args.simulation_alpha is not None:
        if args.simulation_decay is None:
            raise ValueError("--simulation-decay is required with --simulation-alpha")
        simulation_decay = pd.read_csv(args.simulation_decay)
        sim_time = simulation_decay["time_ms"].to_numpy(float)
        sim_signal = simulation_decay["group_weighted_normalized_signal"].to_numpy(float)
        sim_result = invert_single_signal_nnls(
            sim_time,
            sim_signal,
            signal_name="sample16_grouped_2d",
            config=NnlsConfig(
                num_bins=200,
                regularization=args.simulation_alpha,
                t2_min_ms=0.01,
                t2_max_ms=100000.0,
            ),
        )
        sim_t2 = sim_result.t2_bins_ms
        sim_amp = sim_result.spectrum
        pd.DataFrame(
            {
                "t2_ms": sim_t2,
                "amplitude": sim_amp,
                "normalized_amplitude": sim_amp / max(float(sim_amp.max()), 1e-30),
            }
        ).to_csv(
            args.output_dir / f"simulation_t2_inversion_alpha_{args.simulation_alpha:g}.csv",
            index=False,
        )
    else:
        simulation_frame = pd.read_csv(args.simulation_spectrum)
        sim_t2 = simulation_frame["t2_ms"].to_numpy(float)
        sim_amp = simulation_frame[
            "amplitude" if "amplitude" in simulation_frame else "normalized_amplitude"
        ].to_numpy(float)
    sim_valid = (
        np.isfinite(sim_t2)
        & np.isfinite(sim_amp)
        & (sim_t2 > 0)
        & (sim_amp >= 0)
    )
    sim_t2, sim_amp = sim_t2[sim_valid], sim_amp[sim_valid]
    exp_alpha_label = (
        "vendor spectrum"
        if args.experiment_alpha is None
        else rf"$\alpha_{{inv}}={args.experiment_alpha:g}$"
    )
    sim_alpha_label = (
        "precomputed spectrum"
        if args.simulation_alpha is None
        else rf"$\alpha_{{inv}}={args.simulation_alpha:g}$"
    )
    pores = pd.read_csv(args.network_dir / "pores.csv")
    throats = pd.read_csv(args.network_dir / "throats.csv")
    if args.internal_throats_only:
        throats = throats.loc[(throats["pore1_id"] > 0) & (throats["pore2_id"] > 0)]
    pore_radius, pore_volume = positive_arrays(
        1e6 * pores["pore_radius_m"], pores["pore_volume_m3"]
    )
    throat_radius, throat_volume = positive_arrays(
        1e6 * throats["throat_radius_m"], throats["throat_volume_m3"]
    )

    vendor_calibration = calibration(
        pore_radius,
        pore_volume,
        vendor_t2,
        vendor_amp,
        args.ct_porosity,
        args.total_porosity,
        args.alignment_vc,
    )
    reinverted_experiment = calibration(
        pore_radius,
        pore_volume,
        exp_t2,
        exp_amp,
        args.ct_porosity,
        args.total_porosity,
        args.alignment_vc,
    )
    primary = (
        reinverted_experiment
        if args.experiment_alpha is not None and args.rho2_override is None
        else vendor_calibration
    )
    spectrum_porosity = calibration(
        pore_radius,
        pore_volume,
        vendor_t2,
        vendor_amp,
        args.ct_porosity,
        float(vendor_exp_amp.sum()),
        args.alignment_vc,
    )
    combined = calibration(
        pore_radius,
        pore_volume,
        combined_t2,
        combined_amp,
        args.ct_porosity,
        float(combined_amp.sum()),
        args.alignment_vc,
    )
    assert np.isclose(
        3e-3
        * primary["rho2_sphere_alpha3_um_per_s"]
        * primary["nmr_t2_at_alignment_ms"],
        primary["ct_radius_at_alignment_um"],
    )

    pore_order = np.argsort(pore_radius)
    pore_radius_sorted = pore_radius[pore_order]
    pore_cdf_raw = np.cumsum(pore_volume[pore_order]) / pore_volume.sum()
    pore_cdf_adjusted = (
        primary["ct_curve_start_vc"]
        + primary["ct_resolved_fraction"] * pore_cdf_raw
    )
    nmr_order = np.argsort(exp_t2)
    nmr_t2_sorted = exp_t2[nmr_order]
    nmr_cdf = np.cumsum(exp_amp[nmr_order]) / exp_amp.sum()
    calibrated_rho2 = primary["rho2_sphere_alpha3_um_per_s"]
    rho2 = calibrated_rho2 if args.rho2_override is None else args.rho2_override
    nmr_radius_sphere = 0.003 * rho2 * nmr_t2_sorted
    mapped_nmr_radius = 0.003 * rho2 * reinverted_experiment["nmr_t2_at_alignment_ms"]

    pore_diameter = 2.0 * pore_radius
    throat_diameter = 2.0 * throat_radius
    edges = np.geomspace(
        0.8 * min(float(pore_diameter.min()), float(throat_diameter.min())),
        1.2 * max(float(pore_diameter.max()), float(throat_diameter.max())),
        args.histogram_bins + 1,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_hist, _ = np.histogram(pore_diameter, bins=edges, weights=pore_volume)
    throat_hist, _ = np.histogram(throat_diameter, bins=edges, weights=throat_volume)
    pore_hist_norm = pore_hist / pore_hist.max()
    throat_hist_norm = throat_hist / throat_hist.max()
    exp_norm = exp_plot_amp / exp_plot_amp.max()
    sim_norm = sim_amp / sim_amp.max()
    raw_ct_y = args.raw_y or args.raw_ct_y
    pore_plot = pore_hist * 1e18 if raw_ct_y else pore_hist_norm
    throat_plot = throat_hist * 1e18 if raw_ct_y else throat_hist_norm
    exp_plot = exp_plot_amp if args.raw_y else exp_norm
    sim_plot = sim_amp if args.raw_y else sim_norm
    exp_peak_t2 = float(exp_t2[np.argmax(exp_amp)])
    sim_peak_t2 = float(sim_t2[np.argmax(sim_amp)])
    pore_peak = float(centers[np.argmax(pore_hist)])
    throat_peak = float(centers[np.argmax(throat_hist)])
    sphere_peak = float(t2_to_diameter_um(exp_peak_t2, rho2, 3.0))
    cylinder_peak = float(t2_to_diameter_um(exp_peak_t2, rho2, 2.0))
    sim_sphere_peak = float(t2_to_diameter_um(sim_peak_t2, rho2, 3.0))
    previous_sphere_peak = float(t2_to_diameter_um(exp_peak_t2, 4.21, 3.0))

    summaries = []
    for label, result in (
        ("primary_workbook_porosity", primary),
        ("reinverted_experiment_workbook_porosity", reinverted_experiment),
        ("sensitivity_original_spectrum_integral", spectrum_porosity),
        ("sensitivity_combined_spectrum_integral", combined),
    ):
        summaries.append({"case": label, **result, "rho2_used_um_per_s": rho2})
    pd.DataFrame(summaries).to_csv(args.output_dir / "calibration_summary.csv", index=False)
    pd.DataFrame(
        {
            "ct_pore_radius_um": pore_radius_sorted,
            "ct_raw_resolved_cdf": pore_cdf_raw,
            "ct_porosity_corrected_vc": pore_cdf_adjusted,
        }
    ).to_csv(args.output_dir / "ct_cumulative_curve.csv", index=False)
    pd.DataFrame(
        {
            "t2_ms": nmr_t2_sorted,
            "amplitude_cdf": nmr_cdf,
            "sphere_radius_um": nmr_radius_sphere,
        }
    ).to_csv(args.output_dir / "nmr_cumulative_curve.csv", index=False)
    pd.DataFrame(
        {
            "diameter_left_um": edges[:-1],
            "diameter_right_um": edges[1:],
            "diameter_center_um": centers,
            "pore_volume_weighted_frequency": pore_hist,
            "pore_normalized_frequency": pore_hist_norm,
            "throat_volume_weighted_frequency": throat_hist,
            "throat_normalized_frequency": throat_hist_norm,
        }
    ).to_csv(args.output_dir / "pore_throat_diameter_histograms.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "legend.frameon": True,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))
    nmr_ax = ax2.twinx() if raw_ct_y else ax2
    if raw_ct_y:
        nmr_ax.spines["right"].set_visible(True)

    ax1.plot(
        pore_radius_sorted,
        pore_cdf_raw,
        color="#7f93b3",
        linewidth=1.4,
        linestyle="--",
        label=f"{args.ct_source_label} resolved pore volume (uncorrected)",
    )
    ax1.plot(
        pore_radius_sorted,
        pore_cdf_adjusted,
        color="#3e628f",
        linewidth=2.2,
        label=f"{args.ct_source_label} pore volume + unresolved-volume correction",
    )
    ax1.plot(
        nmr_radius_sphere,
        nmr_cdf,
        color="#008577",
        linewidth=2.4,
        label=rf"experimental NMR ($\rho_2={rho2:.2f}$ $\mu$m/s, {exp_alpha_label})",
    )
    ax1.axhline(args.alignment_vc, color="#555555", linewidth=0.9, linestyle=":")
    ax1.axvline(
        primary["ct_radius_at_alignment_um"],
        color="#555555",
        linewidth=0.9,
        linestyle=":",
    )
    ax1.scatter(
        [primary["ct_radius_at_alignment_um"]],
        [args.alignment_vc],
        s=42,
        color="#c64f3d",
        zorder=5,
    )
    ax1.scatter([mapped_nmr_radius], [args.alignment_vc], s=35, color="#008577", zorder=5)
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel(r"Pore-body radius, $r_b$ ($\mu$m)")
    ax1.set_ylabel(r"Cumulative pore-volume fraction, $V_c$")
    ax1.set_title(
        "a  Cumulative-volume consistency check"
        if args.rho2_override is not None
        else "a  Cumulative-volume calibration"
    )
    ax1.grid(which="major", alpha=0.18)
    ax1.grid(which="minor", axis="x", alpha=0.07)
    ax1.legend(loc="lower right", fontsize=7.2, framealpha=0.92)
    ax1.text(
        0.04,
        0.94,
        rf"$V_c={args.alignment_vc:g}$: CT $r={primary['ct_radius_at_alignment_um']:.1f}$ $\mu$m"
        "\n"
        rf"NMR-mapped $r={mapped_nmr_radius:.1f}$ $\mu$m"
        "\n"
        rf"fixed $\rho_2={rho2:.2f}$ $\mu$m/s; {exp_alpha_label}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 4},
    )

    if args.reference_right_style:
        ax2.bar(
            edges[:-1],
            pore_plot,
            width=np.diff(edges),
            align="edge",
            color="#f4a3a8",
            alpha=0.50,
            edgecolor="none",
            label=rf"{args.ct_source_label} pore body (peak {pore_peak:.0f} $\mu$m)",
        )
        ax2.bar(
            edges[:-1],
            throat_plot,
            width=np.diff(edges),
            align="edge",
            color="none",
            edgecolor="#df6f73",
            linewidth=0.75,
            alpha=0.85,
            label=rf"{args.ct_source_label} throat (peak {throat_peak:.0f} $\mu$m)",
        )
        nmr_ax.plot(
            t2_to_diameter_um(exp_plot_t2, rho2, 3.0),
            exp_plot,
            color="black",
            linewidth=1.15,
            marker="o",
            markersize=2.7,
            markevery=max(1, len(exp_plot_t2) // 35),
            label=rf"experimental NMR sphere map (peak {sphere_peak:.0f} $\mu$m)",
        )
        nmr_ax.plot(
            t2_to_diameter_um(sim_t2, rho2, 3.0),
            sim_plot,
            color="#555555",
            linewidth=1.25,
            linestyle="--",
            label=(
                rf"grouped 2D NMR (sim. $\rho_2={args.simulation_forward_rho2_um_per_s:.3f}$; "
                rf"$\alpha={args.simulation_alpha:g}$; peak {sim_sphere_peak:.0f} $\mu$m)"
                if args.simulation_forward_rho2_um_per_s is not None
                else rf"grouped 2D NMR sphere map (peak {sim_sphere_peak:.0f} $\mu$m)"
            ),
        )
    else:
        ax2.stairs(pore_plot, edges, color="#4f6f9f", fill=True, alpha=0.12, linewidth=0)
        ax2.stairs(throat_plot, edges, color="#c45d16", fill=True, alpha=0.09, linewidth=0)
        ax2.stairs(
            pore_plot,
            edges,
            color="#4f6f9f",
            linewidth=1.8,
            label=rf"{args.ct_source_label} pore-body diameter (peak {pore_peak:.0f} $\mu$m)",
        )
        ax2.stairs(
            throat_plot,
            edges,
            color="#c45d16",
            linewidth=1.7,
            linestyle="--",
            label=rf"{args.ct_source_label} throat diameter (peak {throat_peak:.0f} $\mu$m)",
        )
        nmr_ax.plot(
            t2_to_diameter_um(exp_plot_t2, 4.21, 3.0),
            exp_plot,
            color="#8f8f8f",
            linewidth=1.4,
            linestyle=":",
            label=rf"previous $\rho_2=4.21$ sphere map (peak {previous_sphere_peak:.0f} $\mu$m)",
        )
        nmr_ax.plot(
            t2_to_diameter_um(exp_plot_t2, rho2, 3.0),
            exp_plot,
            color="#008577",
            linewidth=2.4,
            label=rf"NMR {exp_alpha_label} $\rightarrow$ sphere (peak {sphere_peak:.0f} $\mu$m)",
        )
        nmr_ax.plot(
            t2_to_diameter_um(exp_plot_t2, rho2, 2.0),
            exp_plot,
            color="#7a4f9a",
            linewidth=2.0,
            linestyle="--",
            label=rf"NMR {exp_alpha_label} $\rightarrow$ cylinder (peak {cylinder_peak:.0f} $\mu$m)",
        )
        nmr_ax.plot(
            t2_to_diameter_um(sim_t2, rho2, 3.0),
            sim_plot,
            color="#1f2937",
            linewidth=2.1,
            linestyle="-.",
            label=rf"grouped 2D NMR {sim_alpha_label} $\rightarrow$ sphere (peak {sim_sphere_peak:.0f} $\mu$m)",
        )
    ax2.set_xscale("log")
    ax2.set_xlim(0.02, args.diameter_max_um)
    ax2.set_xlabel(r"Diameter, $d$ ($\mu$m)")
    if raw_ct_y:
        ax2.set_ylim(bottom=0)
        nmr_ax.set_ylim(bottom=0)
        ax2.set_ylabel(r"CT volume per bin ($\mu$m$^3$)")
        nmr_ax.set_ylabel(
            "NMR spectral amplitude (raw)"
            if args.raw_y
            else "Normalized NMR amplitude"
        )
    else:
        ax2.set_ylim(-0.02, 1.08)
        ax2.set_ylabel("Normalized amplitude / frequency")
    ax2.set_title(
        "b  CT volume and NMR distributions at fixed $\\rho_2$"
        if raw_ct_y
        else "b  Size distributions at fixed $\\rho_2$"
        if args.rho2_override is not None
        else "b  Size distributions after calibration"
    )
    if args.reference_right_style:
        ax2.grid(False)
        ax2.spines["top"].set_visible(True)
        ax2.spines["left"].set_color("#e5484d")
        ax2.tick_params(axis="y", colors="#e5484d")
        ax2.yaxis.label.set_color("#e5484d")
        nmr_ax.spines["top"].set_visible(True)
    else:
        ax2.grid(which="major", alpha=0.18)
        ax2.grid(which="minor", axis="x", alpha=0.07)
    ct_handles, ct_labels = ax2.get_legend_handles_labels()
    nmr_handles, nmr_labels = nmr_ax.get_legend_handles_labels()
    ax2.legend(
        ct_handles + ([] if nmr_ax is ax2 else nmr_handles),
        ct_labels + ([] if nmr_ax is ax2 else nmr_labels),
        loc="upper left" if args.reference_right_style else "upper center",
        bbox_to_anchor=None if args.reference_right_style else (0.5, -0.18),
        fontsize=7.0,
        framealpha=0.0 if args.reference_right_style else 0.92,
        ncol=1 if args.reference_right_style else 2,
    )

    fig.suptitle(
        (
            r"Sample 16: fixed surface relaxivity with separate T2 regularization"
            if args.rho2_override is not None
            else rf"Sample 16 experimental surface-relaxivity calibration ($V_c={args.alignment_vc:g}$)"
        ),
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0 if args.reference_right_style else 0.09, 1, 1))
    figure_path = args.output_dir / args.figure_name
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "method": f"Cumulative-volume alignment at Vc={args.alignment_vc:g}; experimental and simulated spectra retain their selected inversion settings.",
        "calibration_spectrum": "reinverted_experiment" if primary is reinverted_experiment else "vendor_experiment",
        "rho2_used_um_per_s": rho2,
        "simulation_forward_rho2_um_per_s": args.simulation_forward_rho2_um_per_s,
        "rho2_implied_by_reinverted_experiment_um_per_s": reinverted_experiment["rho2_sphere_alpha3_um_per_s"],
        "reinverted_experiment": reinverted_experiment,
        "inversion_regularization": {
            "experimental_alpha": args.experiment_alpha,
            "simulation_alpha": args.simulation_alpha,
        },
        "primary": primary,
        "sensitivity_original_spectrum_integral": spectrum_porosity,
        "sensitivity_combined_spectrum_integral": combined,
        "geometry": {
            "sphere": "r=3*rho2*T2; recommended for pore bodies",
            "cylinder": "r=2*rho2*T2; literal capillary model in Zhang et al.",
        },
        "peak_metrics": {
            "experimental_t2_peak_ms": exp_peak_t2,
            "nmr_sphere_peak_diameter_um_at_primary_rho2": sphere_peak,
            "nmr_cylinder_peak_diameter_um_at_primary_rho2": cylinder_peak,
            "rerun_simulated_t2_peak_ms": sim_peak_t2,
            "rerun_simulated_sphere_peak_diameter_um": sim_sphere_peak,
            "nmr_sphere_peak_diameter_um_at_previous_rho2_4p21": previous_sphere_peak,
            "pnm_pore_peak_diameter_um": pore_peak,
            "pnm_throat_peak_diameter_um": throat_peak,
        },
        "inputs": {
            "experimental_spectrum": str(args.experiment.resolve()),
            "combined_spectrum": str(args.combined.resolve()),
            "rerun_simulation_spectrum": str(
                (args.output_dir / f"simulation_t2_inversion_alpha_{args.simulation_alpha:g}.csv").resolve()
                if args.simulation_alpha is not None else args.simulation_spectrum.resolve()
            ),
            "experimental_decay": str(args.experiment_decay.resolve()) if args.experiment_decay else None,
            "simulation_decay": str(args.simulation_decay.resolve()) if args.simulation_decay else None,
            "pores_csv": str((args.network_dir / "pores.csv").resolve()),
            "throats_csv": str((args.network_dir / "throats.csv").resolve()),
            "throat_scope": "internal only" if args.internal_throats_only else "all source rows",
            "throats_used": int(len(throats)),
            "ct_porosity_pct": args.ct_porosity,
            "total_porosity_pct": args.total_porosity,
            "diameter_display_max_um": args.diameter_max_um,
            "alignment_vc": args.alignment_vc,
            "ct_source_label": args.ct_source_label,
            "y_axis_mode": (
                "raw_all"
                if args.raw_y
                else "raw_ct_normalized_nmr"
                if args.raw_ct_y
                else "normalized"
            ),
            "right_panel_style": "reference_bars_and_marker_lines" if args.reference_right_style else "default",
            "raw_amplitude_maxima": {
                "experimental": float(exp_plot_amp.max()),
                "simulation": float(sim_amp.max()),
            },
        },
        "limitations": [
            f"{args.ct_source_label} equivalent pore-body radii and volumes are used as the micro-CT DIA proxy.",
            "The calibrated value is an effective rho2 under the fast-diffusion, surface-relaxation-dominated model.",
            f"Vc={args.alignment_vc:g} alignment does not force differential-distribution peaks to coincide.",
            "When rho2 is overridden, the selected Vc alignment is shown as a consistency check rather than re-enforced calibration.",
        ],
        "figure": str(figure_path.resolve()),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
