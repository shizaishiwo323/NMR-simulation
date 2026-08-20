"""Plot an NMR T2 spectrum against physically mapped spherical pore bodies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path("simulation_outputs/niu2020_berea_porosity_grouped_nmr_mesh96_alpha0p1_quality0")
PORES_CSV = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\results\niu2020_berea_reproduction3"
    r"\pore_network\pnextract\network_parsed\pores.csv"
)
RHO2_UM_PER_MS = 0.005
T2_BULK_MS = 3000.0
GEOMETRY_FACTOR = 6.0  # sphere expressed by diameter: S/V = 6/D
RUNTIME_RHO2_UM_PER_MS = RHO2_UM_PER_MS
RUNTIME_T2_BULK_MS = T2_BULK_MS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--pores", type=Path, default=PORES_CSV)
    parser.add_argument("--title", default="Berea mesh-96: physical spherical-pore mapping")
    parser.add_argument("--rho2-um-per-ms", type=float, default=RHO2_UM_PER_MS)
    parser.add_argument("--t2-bulk-ms", type=float, default=T2_BULK_MS)
    parser.add_argument("--x-min-ms", type=float, default=100.0)
    parser.add_argument("--x-max-ms", type=float, default=2000.0)
    return parser.parse_args()


def diameter_to_t2(diameter_um: np.ndarray | float) -> np.ndarray:
    diameter_um = np.asarray(diameter_um, dtype=float)
    return 1.0 / (1.0 / RUNTIME_T2_BULK_MS + GEOMETRY_FACTOR * RUNTIME_RHO2_UM_PER_MS / diameter_um)


def t2_to_diameter(t2_ms: np.ndarray | float) -> np.ndarray:
    t2_ms = np.asarray(t2_ms, dtype=float)
    denominator = 1.0 / t2_ms - 1.0 / RUNTIME_T2_BULK_MS
    return GEOMETRY_FACTOR * RUNTIME_RHO2_UM_PER_MS / denominator


def main() -> None:
    global RUNTIME_RHO2_UM_PER_MS, RUNTIME_T2_BULK_MS
    args = parse_args()
    RUNTIME_RHO2_UM_PER_MS = args.rho2_um_per_ms
    RUNTIME_T2_BULK_MS = args.t2_bulk_ms
    lcurve_dir = args.run_dir / "lcurve_inversion"
    spectrum_path = lcurve_dir / "lcurve_optimal_t2_spectrum.csv"
    spectrum = pd.read_csv(spectrum_path)
    lcurve_summary = json.loads((lcurve_dir / "lcurve_summary.json").read_text(encoding="utf-8"))
    pores = pd.read_csv(args.pores)

    diameter_um = 2.0 * pores["pore_radius_m"].to_numpy(float) * 1e6
    volume_m3 = pores["pore_volume_m3"].to_numpy(float)
    valid = np.isfinite(diameter_um) & np.isfinite(volume_m3) & (diameter_um > 0) & (volume_m3 > 0)
    diameter_um, volume_m3 = diameter_um[valid], volume_m3[valid]

    edges_um = np.geomspace(diameter_um.min(), diameter_um.max(), 81)
    histogram, _ = np.histogram(diameter_um, bins=edges_um, weights=volume_m3)
    histogram = histogram / histogram.max()
    centers_um = np.sqrt(edges_um[:-1] * edges_um[1:])
    mapped_edges_ms = diameter_to_t2(edges_um)
    mapped_centers_ms = diameter_to_t2(centers_um)

    pore_peak_index = int(np.argmax(histogram))
    pore_peak_um = float(centers_um[pore_peak_index])
    pore_peak_t2_ms = float(mapped_centers_ms[pore_peak_index])
    t2_peak_index = int(spectrum["amplitude"].idxmax())
    t2_peak_ms = float(spectrum.loc[t2_peak_index, "t2_ms"])
    t2_peak_diameter_um = float(t2_to_diameter(t2_peak_ms))

    histogram_path = lcurve_dir / "spherical_pore_diameter_log_volume_weighted_histogram.csv"
    pd.DataFrame(
        {
            "pore_diameter_left_um": edges_um[:-1],
            "pore_diameter_right_um": edges_um[1:],
            "pore_diameter_center_um": centers_um,
            "normalized_volume_weighted_frequency": histogram,
            "mapped_t2_left_ms": mapped_edges_ms[:-1],
            "mapped_t2_right_ms": mapped_edges_ms[1:],
            "mapped_t2_center_ms": mapped_centers_ms,
        }
    ).to_csv(histogram_path, index=False)

    output_path = lcurve_dir / "lcurve_t2_vs_spherical_pore_physical_log.png"
    manifest_path = lcurve_dir / "t2_spherical_pore_physical_log_manifest.json"
    manifest = {
        "t2_spectrum": str(spectrum_path.resolve()),
        "pore_table": str(args.pores.resolve()),
        "pore_radius_field": "pore_radius_m",
        "pore_count": int(valid.sum()),
        "excluded_nonpositive_or_nonfinite_pores": int((~valid).sum()),
        "weighting": "pore_volume_m3",
        "binning": "80 logarithmic pore-diameter bins",
        "physical_mapping": "1/T2 = 1/T2_bulk + 6*rho2/D (spherical pore body)",
        "rho2_um_per_ms": RUNTIME_RHO2_UM_PER_MS,
        "t2_bulk_ms": RUNTIME_T2_BULK_MS,
        "geometry_factor_for_diameter": GEOMETRY_FACTOR,
        "t2_peak_ms": t2_peak_ms,
        "lcurve_alpha": float(lcurve_summary["best_alpha"]),
        "t2_peak_equivalent_pore_diameter_um": t2_peak_diameter_um,
        "pore_diameter_distribution_peak_um": pore_peak_um,
        "pore_peak_mapped_t2_ms": pore_peak_t2_ms,
        "axes": "bottom T2 log axis; top physically coupled pore-diameter log axis",
        "alignment": "physical mapping only; no display shifting or peak forcing",
        "histogram_csv": str(histogram_path.resolve()),
        "output_png": str(output_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    pore_line = ax.stairs(
        histogram,
        mapped_edges_ms,
        color="#52789c",
        lw=2.0,
        label="volume-weighted spherical pore diameter",
    )
    t2_line, = ax.plot(
        spectrum["t2_ms"],
        spectrum["normalized_amplitude"],
        color="#20242b",
        lw=2.3,
        label=rf"L-curve $T_2$ spectrum ($\alpha={lcurve_summary['best_alpha']:.3g}$)",
    )
    ax.axvline(t2_peak_ms, color="#c54f3a", lw=1.1, ls="--")
    ax.axvline(pore_peak_t2_ms, color="#52789c", lw=1.1, ls=":")
    ax.scatter([t2_peak_ms], [1.0], color="#c54f3a", s=26, zorder=4)
    ax.scatter([pore_peak_t2_ms], [1.0], color="#52789c", s=26, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(args.x_min_ms, args.x_max_ms)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"$T_2$ (ms)")
    ax.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax.grid(which="major", alpha=0.18)
    ax.legend([pore_line, t2_line], [pore_line.get_label(), t2_line.get_label()], loc="upper left")
    fig.suptitle(args.title, fontweight="bold")

    top = ax.secondary_xaxis("top", functions=(t2_to_diameter, diameter_to_t2))
    top.set_xscale("log")
    top.set_xlabel(r"pore diameter ($\mu$m), sphere model")
    top.set_xticks([5, 10, 20, 50, 100, 200])
    top.set_xticklabels(["5", "10", "20", "50", "100", "200"])

    ax.text(
        0.05,
        0.58,
        f"NMR peak: {t2_peak_ms:.0f} ms = {t2_peak_diameter_um:.1f} $\mu$m\n"
        f"PNM peak: {pore_peak_um:.1f} $\mu$m = {pore_peak_t2_ms:.0f} ms",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.25",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    assert np.allclose(t2_to_diameter(diameter_to_t2(centers_um)), centers_um)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
