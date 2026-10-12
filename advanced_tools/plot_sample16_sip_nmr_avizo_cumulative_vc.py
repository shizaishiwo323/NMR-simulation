"""Plot Sample 16 SIP, experimental NMR, and Avizo CT cumulative pore size."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.compare_experimental_t2_with_sim_average import load_experiment_decay
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = ROOT / "simulation_outputs/sample_16_avizo_ct_rho2_14p63_same_nmr_20260916"
PNEXTRACT_CT_DIR = ROOT / "simulation_outputs/sample_16_pnextract_ct_vc1_reference_right_style_20260916"
SIP_CSV = Path(
    r"E:\python\jupyter\codex\sip模拟\results\sample16_sip_pore_distribution_20260915"
    r"\source_data\sip_equivalent_pore_throat_distribution.csv"
)
AVIZO_WORKBOOK = Path(
    r"E:\python\jupyter\codex\sip模拟\data\ct_backed_samples_raw_copy_20260605"
    r"\sample_16_Grainstone\CT_slices\avizo\16-avizo-pore throat.xlsx"
)
NMR_SPECTRUM = Path(
    r"E:\python\jupyter\codex\sip模拟\data\ct_backed_samples_raw_copy_20260605"
    r"\sample_16_Grainstone\NMR_original\16-1\T2CPMG\Spectra\spectrum.csv"
)
NMR_DECAY = NMR_SPECTRUM.parents[1] / "data.csv"
OUTPUT_DIR = ROOT / "simulation_outputs/sample_16_trimmed_sequential_alignment_20260916"
SIP_SUPPORTED_RADIUS_UM = (0.114, 3.478)
SIP_MIN_ACTIVE_MASS_FRACTION = 1e-4
CT_MIN_VISIBLE_GROWTH_VC = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-source", choices=("avizo", "pnextract"), default="avizo")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--experiment-alpha", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.experiment_alpha is not None and args.experiment_alpha < 0:
        raise ValueError("--experiment-alpha must be nonnegative")
    trimmed = not args.no_trim
    ct_dir = PNEXTRACT_CT_DIR if args.ct_source == "pnextract" else CALIBRATION_DIR
    ct_label = "pnextract CT pore body" if args.ct_source == "pnextract" else "Avizo CT pore body"
    output_dir = (
        ROOT / (
            "simulation_outputs/sample_16_pnextract_sequential_alignment_after_trim_20260916"
            if trimmed else "simulation_outputs/sample_16_pnextract_sequential_alignment_before_trim_20260916"
        )
        if args.ct_source == "pnextract" else OUTPUT_DIR
    )
    if args.output_dir is not None:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_source_path = ct_dir / "ct_cumulative_curve.csv"
    ct = pd.read_csv(ct_source_path)
    nmr_spectrum_path = NMR_SPECTRUM
    if args.experiment_alpha is None:
        nmr = pd.read_csv(CALIBRATION_DIR / "nmr_cumulative_curve.csv")
    else:
        time_ms, signal, _ = load_experiment_decay(NMR_DECAY)
        inversion = invert_single_signal_nnls(
            time_ms, signal, signal_name="sample16_experiment",
            config=NnlsConfig(num_bins=200, regularization=args.experiment_alpha,
                              t2_min_ms=0.01, t2_max_ms=100000.0),
        )
        spectrum = pd.DataFrame({"t2_ms": inversion.t2_bins_ms,
                                 "amplitude": inversion.spectrum})
        nmr_spectrum_path = output_dir / f"experiment_t2_inversion_alpha_{args.experiment_alpha:g}.csv"
        spectrum.to_csv(nmr_spectrum_path, index=False)
        spectrum = spectrum[spectrum["amplitude"] > 0].copy()
        nmr = pd.DataFrame({"t2_ms": spectrum["t2_ms"],
                            "amplitude_cdf": spectrum["amplitude"].cumsum() / spectrum["amplitude"].sum()})
    sip = pd.read_csv(SIP_CSV).sort_values("pore_throat_radius_um")
    calibration = json.loads((CALIBRATION_DIR / "run_manifest.json").read_text(encoding="utf-8"))

    ct_original_min_radius = float(ct["ct_pore_radius_um"].min())
    ct_min_vc = float(ct["ct_porosity_corrected_vc"].iloc[0])
    ct_growth_threshold = ct_min_vc + CT_MIN_VISIBLE_GROWTH_VC
    if trimmed:
        ct_min_radius = float(
            ct.loc[ct["ct_porosity_corrected_vc"] >= ct_growth_threshold, "ct_pore_radius_um"].iloc[0]
        )
        ct = ct[ct["ct_pore_radius_um"] >= ct_min_radius].copy()
    ct_max_radius = float(ct["ct_pore_radius_um"].max())

    rho2_base = (
        float(calibration["rho2_implied_by_reinverted_experiment_um_per_s"])
        if args.experiment_alpha is None
        else ct_max_radius / (0.003 * float(nmr["t2_ms"].max()))
    )
    nmr_radius_base = 0.003 * rho2_base * nmr["t2_ms"].to_numpy(float)
    nmr_scale = ct_max_radius / float(nmr_radius_base.max())
    nmr_radius = nmr_scale * nmr_radius_base
    rho2_display = nmr_scale * rho2_base
    sip = sip[sip["pore_throat_radius_um"].between(*SIP_SUPPORTED_RADIUS_UM)].copy()
    sip_mass_fraction = (
        sip["normalized_pore_volume_fraction_per_log_bin"]
        / sip["normalized_pore_volume_fraction_per_log_bin"].sum()
    )
    if trimmed:
        sip_last_active_index = sip.index[sip_mass_fraction >= SIP_MIN_ACTIVE_MASS_FRACTION][-1]
        sip = sip.loc[:sip_last_active_index].copy()
    sip_mass = sip["normalized_pore_volume_fraction_per_log_bin"].to_numpy(float)
    sip_shape = np.cumsum(sip_mass) / sip_mass.sum()
    ct_for_interp = ct.groupby("ct_pore_radius_um", as_index=False)[
        "ct_porosity_corrected_vc"
    ].max()
    sip_max_radius = float(sip["pore_throat_radius_um"].max())
    sip_anchor_below_ct_support = sip_max_radius < float(ct_for_interp["ct_pore_radius_um"].min())
    sip_anchor_vc = float(
        np.interp(
            np.log10(sip_max_radius),
            np.log10(ct_for_interp["ct_pore_radius_um"]),
            ct_for_interp["ct_porosity_corrected_vc"],
        )
    )
    sip_vc = sip_anchor_vc * sip_shape
    nmr_radius_at_sip_anchor = float(
        10
        ** np.interp(
            sip_anchor_vc,
            nmr["amplitude_cdf"],
            np.log10(nmr_radius),
        )
    )
    sip_radius_scale = nmr_radius_at_sip_anchor / sip_max_radius
    sip_radius = sip_radius_scale * sip["pore_throat_radius_um"].to_numpy(float)

    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": ct_label,
                    "original_radius_um": ct["ct_pore_radius_um"],
                    "radius_um": ct["ct_pore_radius_um"],
                    "cumulative_vc": ct["ct_porosity_corrected_vc"],
                    "support": (
                        "CT low-radius plateau removed at 0.01 Vc growth above the unresolved-volume baseline"
                        if trimmed else "full resolved CT range"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "method": "NMR experiment pore body",
                    "original_radius_um": nmr_radius_base,
                    "radius_um": nmr_radius,
                    "cumulative_vc": nmr["amplitude_cdf"],
                    "support": "r = 3 rho2 T2",
                }
            ),
            pd.DataFrame(
                {
                    "method": "SIP equivalent pore throat",
                    "original_radius_um": sip["pore_throat_radius_um"],
                    "radius_um": sip_radius,
                    "cumulative_vc": sip_vc,
                    "support": (
                        "measured-frequency-supported active range; terminal near-zero-mass bins removed"
                        if trimmed else "full measured-frequency-supported range"
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    curves.to_csv(output_dir / "sample16_trimmed_sequential_alignment_source_data.csv", index=False)

    assert all(
        group["cumulative_vc"].is_monotonic_increasing
        for _, group in curves.groupby("method")
    )
    endpoints = curves.groupby("method")["cumulative_vc"].max()
    assert np.isclose(endpoints[ct_label], 1.0)
    assert np.isclose(endpoints["NMR experiment pore body"], 1.0)
    assert np.isclose(endpoints["SIP equivalent pore throat"], sip_anchor_vc)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.1))
    ax.plot(
        ct["ct_pore_radius_um"],
        ct["ct_porosity_corrected_vc"],
        color="#184f59",
        linewidth=2.0,
        label=rf"{args.ct_source} $\mu$-CT (pore body)",
    )
    ax.plot(
        nmr_radius,
        nmr["amplitude_cdf"],
        color="#234dcc",
        linewidth=2.0,
        label=rf"NMR ($\alpha={args.experiment_alpha:g}$; $\rho_2^*={rho2_display:.2f}$ $\mu$m s$^{{-1}}$)"
        if args.experiment_alpha is not None
        else rf"NMR ($V_c=1$ aligned to CT; $\rho_2^*={rho2_display:.2f}$ $\mu$m s$^{{-1}}$)",
    )
    ax.plot(
        sip_radius,
        sip_vc,
        color="#b17a2b",
        linewidth=2.2,
        label=rf"SIP max aligned to NMR at $V_c={sip_anchor_vc:.3f}$",
    )

    ax.set_xscale("log")
    ax.set_xlim(min(0.01, float(nmr_radius.min()) * 0.8), 300)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Aligned equivalent pore radius, $r$ ($\mu$m)")
    ax.set_ylabel(r"Normalized cumulative pore volume, $V_c$")
    ax.set_title(f"Sample 16 pore-size comparison ({'after' if trimmed else 'before'} trimming)", pad=10)
    ax.grid(which="major", color="#333333", linewidth=0.75, alpha=0.62)
    ax.grid(which="minor", color="#777777", linewidth=0.45, alpha=0.34)
    ax.tick_params(which="both", direction="out", top=True, right=True, length=4)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.03, 0.55),
        frameon=True,
        framealpha=0.96,
        edgecolor="#333333",
        fontsize=8,
    )
    ax.text(
        0.03,
        0.97,
        rf"{args.ct_source} CT: {ct['ct_pore_radius_um'].min():.2f}$-${ct_max_radius:.2f} $\mu$m; NMR $V_c=1$ aligned"
        "\n"
        rf"SIP: {sip['pore_throat_radius_um'].min():.3f}$-${sip_max_radius:.3f} $\mu$m; max $\rightarrow$ {nmr_radius_at_sip_anchor:.1f} $\mu$m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    fig.tight_layout()
    figure_path = output_dir / (
        f"sample16_pnextract_sequential_alignment_{'after' if trimmed else 'before'}_trim.png"
        if args.ct_source == "pnextract" else "sample16_trimmed_sequential_alignment.png"
    )
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "figure": str(figure_path.resolve()),
        "ct_source": args.ct_source,
        "trimmed": trimmed,
        "vertical_axis": "linear Vc from 0 to 1",
        "endpoint_calibration": {
            "NMR_max_vc": 1.0,
            "CT_max_vc": 1.0,
            "CT_min_vc": float(ct["ct_porosity_corrected_vc"].iloc[0]),
            "SIP_max_radius_um": sip_max_radius,
            "SIP_max_vc_from_CT_curve": sip_anchor_vc,
            "SIP_anchor_below_CT_radius_support": sip_anchor_below_ct_support,
        },
        "nmr_radius_mapping": {
            "formula": "r = 3 rho2 T2",
            "experimental_inversion_alpha": args.experiment_alpha,
            "experimental_inversion_input": "original CPMG decay" if args.experiment_alpha is not None else "vendor spectrum",
            "base_rho2_um_per_s": rho2_base,
            "display_aligned_rho2_um_per_s": rho2_display,
            "horizontal_scale_factor": nmr_scale,
            "alignment": "NMR maximum-radius point at Vc=1 aligned to the selected CT maximum-radius point at Vc=1",
        },
        "sip": {
            "vertical_calibration": "Shape normalized within the measured-frequency-supported interval, then multiplied by the selected CT Vc at the SIP maximum radius (left-edge CT baseline when below CT support).",
            "equivalent_geometry": "pore throat",
            "measured_frequency_supported_radius_um": list(SIP_SUPPORTED_RADIUS_UM),
            "horizontal_scale_factor": sip_radius_scale,
            "display_aligned_max_radius_um": nmr_radius_at_sip_anchor,
            "alignment": "SIP maximum point aligned to the already CT-aligned NMR curve at the same Vc",
        },
        "trimming": {
            "CT_original_min_radius_um": ct_original_min_radius,
            "CT_retained_radius_range_um": [
                float(ct["ct_pore_radius_um"].min()),
                ct_max_radius,
            ],
            "CT_removed_low_radius_resolved_volume_fraction": (
                float(ct["ct_raw_resolved_cdf"].iloc[0]) if trimmed else 0.0
            ),
            "CT_rule": (
                f"Remove the low-radius plateau until corrected Vc grows by {CT_MIN_VISIBLE_GROWTH_VC:g} above its unresolved-volume baseline."
                if trimmed else "No CT trimming."
            ),
            "SIP_supported_original_radius_range_um": list(SIP_SUPPORTED_RADIUS_UM),
            "SIP_retained_radius_range_um": [
                float(sip["pore_throat_radius_um"].min()),
                sip_max_radius,
            ],
            "SIP_rule": (
                f"Remove terminal bins with support-normalized mass fraction below {SIP_MIN_ACTIVE_MASS_FRACTION:g}."
                if trimmed else "No trimming within the measured-frequency-supported interval."
            ),
        },
        "inputs": {
            "ct_source": args.ct_source,
            "ct_cumulative_curve": str(ct_source_path.resolve()),
            "ct_cumulative_curve_sha256": sha256(ct_source_path),
            "ct_extraction_manifest": str((ct_dir / "run_manifest.json").resolve()),
            "nmr_experimental_spectrum": str(nmr_spectrum_path.resolve()),
            "nmr_experimental_spectrum_sha256": sha256(nmr_spectrum_path),
            "nmr_experimental_decay": str(NMR_DECAY.resolve()) if args.experiment_alpha is not None else None,
            "nmr_experimental_decay_sha256": sha256(NMR_DECAY) if args.experiment_alpha is not None else None,
            "sip_distribution": str(SIP_CSV),
            "sip_distribution_sha256": sha256(SIP_CSV),
        },
        "scientific_boundary": "The horizontal NMR and SIP shifts are user-requested display alignments. The aligned radii are not independent physical pore-size determinations; the paper instead aligns CT/NMR in an overlapping range and SIP/MIP at the SIP resolution limit.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
