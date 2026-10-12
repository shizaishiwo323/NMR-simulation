"""Plot one grouped-NMR T2 spectrum against Avizo CT with bulk-T2 correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spectrum", type=Path)
    source.add_argument("--decay", type=Path)
    parser.add_argument("--avizo-histogram", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rho2-um-per-s", type=float, default=29.863969599115315)
    parser.add_argument("--bulk-t2-ms", type=float, default=3000.0)
    parser.add_argument("--inversion-alpha", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rho2_um_per_s <= 0 or args.bulk_t2_ms <= 0:
        raise ValueError("rho2 and bulk T2 must be positive")
    if args.decay is not None:
        decay = pd.read_csv(args.decay)
        result = invert_single_signal_nnls(
            decay["time_ms"].to_numpy(float),
            decay["group_weighted_normalized_signal"].to_numpy(float),
            signal_name="sample16_grouped_nmr",
            config=NnlsConfig(num_bins=200, regularization=args.inversion_alpha,
                              t2_min_ms=0.01, t2_max_ms=100000.0),
        )
        spectrum = pd.DataFrame({"t2_ms": result.t2_bins_ms, "amplitude": result.spectrum})
    else:
        spectrum = pd.read_csv(args.spectrum)
    t2_ms = spectrum["t2_ms"].to_numpy(float)
    amplitude = spectrum["amplitude"].to_numpy(float)
    if not (np.all(np.isfinite(t2_ms)) and np.all(np.isfinite(amplitude))
            and np.all(t2_ms > 0) and np.all(amplitude >= 0) and amplitude.sum() > 0):
        raise ValueError("Invalid T2 spectrum")

    valid = t2_ms < args.bulk_t2_ms
    diameter_um = np.full_like(t2_ms, np.nan)
    diameter_um[valid] = (
        0.006 * args.rho2_um_per_s * t2_ms[valid]
        / (1.0 - t2_ms[valid] / args.bulk_t2_ms)
    )
    density_per_log_d = np.full_like(amplitude, np.nan)
    density_per_log_d[valid] = amplitude[valid] * (1.0 - t2_ms[valid] / args.bulk_t2_ms)
    density_normalized = density_per_log_d / np.nanmax(density_per_log_d)
    mapped = spectrum[["t2_ms", "amplitude"]].copy()
    mapped["equivalent_diameter_um"] = diameter_um
    mapped["finite_diameter"] = valid
    mapped["amplitude_fraction_of_total"] = amplitude / amplitude.sum()
    mapped["density_per_log_d"] = density_per_log_d
    mapped["normalized_density_per_log_d"] = density_normalized

    hist = pd.read_csv(args.avizo_histogram)
    ct_edges = np.r_[hist["diameter_left_um"].to_numpy(float), hist["diameter_right_um"].iloc[-1]]
    ratio = ct_edges[1] / ct_edges[0]
    edges = ct_edges.copy()
    while edges[-1] < 2000:
        edges = np.r_[edges, edges[-1] * ratio]
    centers = np.sqrt(edges[:-1] * edges[1:])
    ct_pore_volume = np.pad(hist["pore_volume_weighted_frequency"].to_numpy(float) * 1e18,
                            (0, len(centers) - len(hist)))
    ct_throat_volume = np.pad(hist["throat_volume_weighted_frequency"].to_numpy(float) * 1e18,
                              (0, len(centers) - len(hist)))
    nmr_mass, _ = np.histogram(diameter_um[valid], bins=edges, weights=amplitude[valid])
    nmr_peak = float(diameter_um[valid][np.argmax(density_per_log_d[valid])])
    ct_pore_peak = float(centers[np.argmax(ct_pore_volume)])
    ct_throat_peak = float(centers[np.argmax(ct_throat_volume)])
    unmappable_fraction = float(amplitude[~valid].sum() / amplitude.sum())
    beyond_plot_fraction = float(amplitude[valid & (diameter_um > 2000)].sum() / amplitude.sum())
    binned = pd.DataFrame({
        "diameter_left_um": edges[:-1], "diameter_right_um": edges[1:],
        "diameter_center_um": centers, "avizo_pore_volume_um3": ct_pore_volume,
        "avizo_throat_volume_um3": ct_throat_volume,
        "nmr_amplitude_per_log_d_bin": nmr_mass,
        "nmr_fraction_of_total": nmr_mass / amplitude.sum(),
        "nmr_normalized_amplitude": nmr_mass / nmr_mass.max(),
    })

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(7.7, 5.1))
    nmr_ax = ax.twinx()
    ax.bar(edges[:-1], ct_pore_volume, width=np.diff(edges), align="edge",
           color="#f4a3a8", alpha=0.46, edgecolor="none",
           label=rf"Avizo CT pore body (peak {ct_pore_peak:.0f} $\mu$m)")
    ax.bar(edges[:-1], ct_throat_volume, width=np.diff(edges), align="edge",
           color="none", edgecolor="#d9757c", linewidth=0.8, alpha=0.8,
           label=rf"Avizo CT throat (peak {ct_throat_peak:.0f} $\mu$m)")
    nmr_ax.plot(diameter_um[valid], density_normalized[valid], color="#2453a6",
                linewidth=2.1, label=rf"grouped-2D NMR (peak {nmr_peak:.0f} $\mu$m)")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 2000)
    ax.set_ylim(bottom=0)
    nmr_ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Equivalent diameter, $d$ ($\mu$m)")
    ax.set_ylabel(r"Avizo CT volume per bin ($\mu$m$^3$)", color="#d95259")
    nmr_ax.set_ylabel("Normalized NMR density per log diameter")
    ax.tick_params(axis="y", colors="#d95259")
    ax.spines["left"].set_color("#d95259")
    ax.set_title(r"Sample 16: CT vs NMR with bulk-$T_2$ correction")
    ax.text(0.98, 0.06,
            rf"$\rho_2={args.rho2_um_per_s:.3f}$ $\mu$m/s; "
            rf"$T_{{2B}}={args.bulk_t2_ms / 1000:.1f}$ s; "
            rf"$\alpha={args.inversion_alpha:g}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    handles_ct, labels_ct = ax.get_legend_handles_labels()
    handles_nmr, labels_nmr = nmr_ax.get_legend_handles_labels()
    ax.legend(handles_ct + handles_nmr, labels_ct + labels_nmr, loc="upper left",
              fontsize=8, frameon=False)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.decay is not None:
        spectrum.to_csv(args.output_dir / f"simulation_t2_inversion_alpha_{args.inversion_alpha:g}.csv", index=False)
    rho_token = f"{args.rho2_um_per_s:.3f}".replace(".", "p")
    figure_path = args.output_dir / f"sample16_rho2_{rho_token}_bulk_corrected_nmr_vs_avizo.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    mapped.to_csv(args.output_dir / "nmr_t2_to_bulk_corrected_diameter.csv", index=False)
    binned.to_csv(args.output_dir / "avizo_and_bulk_corrected_nmr_log_bins.csv", index=False)
    (args.output_dir / "method_manifest.json").write_text(json.dumps({
        "rho2_um_per_s": args.rho2_um_per_s,
        "bulk_t2_ms": args.bulk_t2_ms,
        "inversion_alpha": args.inversion_alpha,
        "model": "spherical equivalent pore body, fast diffusion: 1/T2 = 1/T2B + 6*rho2/d; no separate internal-gradient relaxation term",
        "amplitude_handling": "Original NNLS amplitudes are treated as mass per uniform log-T2 bin. The plotted density per log diameter applies d(log T2)/d(log d) = 1 - T2/T2B; mapped bin masses are also saved for audit. No artificial size is assigned where T2 >= T2B.",
        "unmappable_amplitude_fraction_T2_at_or_above_T2B": unmappable_fraction,
        "mapped_amplitude_fraction_above_plot_2000_um": beyond_plot_fraction,
        "peaks_um": {"nmr_density_per_log_d": nmr_peak, "avizo_pore_body": ct_pore_peak,
                     "avizo_throat": ct_throat_peak},
        "inputs": {"nmr_spectrum": str(args.spectrum.resolve()) if args.spectrum else None,
                   "group_weighted_decay": str(args.decay.resolve()) if args.decay else None,
                   "avizo_histogram": str(args.avizo_histogram.resolve())},
        "figure": str(figure_path.resolve()),
        "model_dimension_note": "Porosity-grouped 2D representative-slice (2.5D) approximation, not a 3D NMR solve.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Figure: {figure_path.resolve()}")
    print(f"Peaks (um): NMR={nmr_peak:.3f}, CT pore={ct_pore_peak:.3f}, CT throat={ct_throat_peak:.3f}")
    print(f"T2>=T2B mass={100*unmappable_fraction:.3f}%; >2000 um mass={100*beyond_plot_fraction:.3f}%")


if __name__ == "__main__":
    main()
