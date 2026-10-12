"""Compare pnextract CT with measured and grouped-slice NMR at one rho2."""

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


def mapped_spectrum(t2_ms: np.ndarray, amplitude: np.ndarray, rho2_um_per_s: float) -> pd.DataFrame:
    """Map equal-log-T2 spectrum bins to spherical equivalent diameter, without bulk T2."""
    if not (np.all(np.isfinite(t2_ms)) and np.all(t2_ms > 0)
            and np.all(np.isfinite(amplitude)) and np.all(amplitude >= 0)
            and np.max(amplitude) > 0):
        raise ValueError("Invalid T2 spectrum")
    return pd.DataFrame({
        "t2_ms": t2_ms,
        "amplitude": amplitude,
        "equivalent_diameter_um": 0.006 * rho2_um_per_s * t2_ms,
        "normalized_density_per_log_d": amplitude / np.max(amplitude),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-decay", type=Path, required=True)
    parser.add_argument("--experimental-spectrum", type=Path, required=True)
    parser.add_argument("--ct-histogram", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rho2-um-per-s", type=float, default=26.4)
    parser.add_argument("--inversion-alpha", type=float, default=500)
    args = parser.parse_args()
    if args.rho2_um_per_s <= 0 or args.inversion_alpha < 0:
        raise ValueError("rho2 must be positive and alpha nonnegative")

    decay = pd.read_csv(args.grouped_decay)
    inversion = invert_single_signal_nnls(
        decay["time_ms"].to_numpy(float),
        decay["group_weighted_normalized_signal"].to_numpy(float),
        signal_name="sample16_grouped_nmr",
        config=NnlsConfig(num_bins=200, regularization=args.inversion_alpha,
                          t2_min_ms=0.01, t2_max_ms=100000.0),
    )
    sim = mapped_spectrum(inversion.t2_bins_ms, inversion.spectrum, args.rho2_um_per_s)
    measured = pd.read_csv(args.experimental_spectrum, header=None, names=["t2_ms", "amplitude"])
    exp = mapped_spectrum(measured.t2_ms.to_numpy(float), measured.amplitude.to_numpy(float),
                          args.rho2_um_per_s)

    ct = pd.read_csv(args.ct_histogram)
    edges = np.r_[ct.diameter_left_um.to_numpy(float), ct.diameter_right_um.iloc[-1]]
    pore_um3 = ct.pore_volume_weighted_frequency.to_numpy(float) * 1e18
    throat_um3 = ct.throat_volume_weighted_frequency.to_numpy(float) * 1e18
    if not (np.all(np.diff(edges) > 0) and np.all(pore_um3 >= 0) and np.all(throat_um3 >= 0)):
        raise ValueError("Invalid CT histogram")
    ct_pore_peak = float(ct.diameter_center_um.iloc[np.argmax(pore_um3)])
    ct_throat_peak = float(ct.diameter_center_um.iloc[np.argmax(throat_um3)])
    sim_peak = float(sim.equivalent_diameter_um.iloc[sim.amplitude.idxmax()])
    exp_peak = float(exp.equivalent_diameter_um.iloc[exp.amplitude.idxmax()])

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    nmr_ax = ax.twinx()
    ax.bar(edges[:-1], pore_um3, width=np.diff(edges), align="edge",
           color="#f4a3a8", alpha=0.46, edgecolor="none",
           label=rf"pnextract CT pore body (peak {ct_pore_peak:.0f} $\mu$m)")
    ax.bar(edges[:-1], throat_um3, width=np.diff(edges), align="edge",
           color="none", edgecolor="#d9757c", linewidth=0.8, alpha=0.8,
           label=rf"pnextract CT throat (peak {ct_throat_peak:.0f} $\mu$m)")
    nmr_ax.plot(exp.equivalent_diameter_um, exp.normalized_density_per_log_d,
                color="black", linewidth=1.7, marker="o", markersize=2.5,
                label=rf"measured NMR (peak {exp_peak:.0f} $\mu$m)")
    nmr_ax.plot(sim.equivalent_diameter_um, sim.normalized_density_per_log_d,
                color="#2453a6", linewidth=2.2,
                label=rf"grouped-slice NMR simulation (peak {sim_peak:.0f} $\mu$m)")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 2000)
    ax.set_ylim(bottom=0)
    nmr_ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Equivalent diameter, $d$ ($\mu$m)")
    ax.set_ylabel(r"pnextract CT volume per bin ($\mu$m$^3$)", color="#d95259")
    nmr_ax.set_ylabel("Normalized NMR density per log diameter")
    ax.tick_params(axis="y", colors="#d95259")
    ax.spines["left"].set_color("#d95259")
    ax.set_title("Sample 16: pnextract CT vs measured and simulated NMR")
    ax.text(0.98, 0.06, rf"$\rho_2={args.rho2_um_per_s:.2f}$ $\mu$m/s; "
            rf"no bulk-$T_2$ term; simulation $\alpha={args.inversion_alpha:g}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ct_handles, ct_labels = ax.get_legend_handles_labels()
    nmr_handles, nmr_labels = nmr_ax.get_legend_handles_labels()
    ax.legend(ct_handles + nmr_handles, ct_labels + nmr_labels,
              loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "sample16_pnextract_ct_measured_simulated_nmr_rho2_26p4_no_bulk.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    exp.to_csv(args.output_dir / "measured_nmr_equivalent_diameter.csv", index=False)
    sim.to_csv(args.output_dir / "simulated_nmr_equivalent_diameter.csv", index=False)
    (args.output_dir / "method_manifest.json").write_text(json.dumps({
        "rho2_um_per_s": args.rho2_um_per_s,
        "bulk_t2_correction": False,
        "diameter_formula_um": "d = 0.006 * rho2[um/s] * T2[ms] (spherical S/V=6/d)",
        "simulated_spectrum_inversion_alpha": args.inversion_alpha,
        "measured_spectrum": "vendor-provided spectrum, not re-inverted",
        "ct_histogram": "pnextract pore and throat volume-weighted bins",
        "nmr_curves": "each independently peak-normalized; amplitudes correspond to equal log-T2 bins",
        "simulation_dimension": "porosity-grouped 2D representative slices (2.5D approximation), not a 3D solve",
        "peaks_um": {"ct_pore_body": ct_pore_peak, "ct_throat": ct_throat_peak,
                     "measured_nmr": exp_peak, "simulated_nmr": sim_peak},
        "inputs": {"grouped_decay": str(args.grouped_decay.resolve()),
                   "experimental_spectrum": str(args.experimental_spectrum.resolve()),
                   "ct_histogram": str(args.ct_histogram.resolve())},
        "figure": str(figure.resolve()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Figure: {figure.resolve()}")
    print(f"Peaks (um): CT pore={ct_pore_peak:.3f}, CT throat={ct_throat_peak:.3f}, "
          f"measured NMR={exp_peak:.3f}, simulated NMR={sim_peak:.3f}")


if __name__ == "__main__":
    assert np.isclose(mapped_spectrum(np.array([1000.]), np.array([1.]), 26.4)
                      .equivalent_diameter_um.iloc[0], 158.4)
    main()
