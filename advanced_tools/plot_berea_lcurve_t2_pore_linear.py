"""Plot L-curve T2 against a linear, volume-weighted pore-size distribution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path("simulation_outputs/niu2020_berea_porosity_grouped_nmr_mesh96_alpha0p1_quality0")
LCURVE_DIR = RUN_DIR / "lcurve_inversion"
PORES_CSV = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\results\niu2020_berea_reproduction3"
    r"\pore_network\pnextract\network_parsed\pores.csv"
)


def main() -> None:
    spectrum = pd.read_csv(LCURVE_DIR / "lcurve_optimal_t2_spectrum.csv")
    pores = pd.read_csv(PORES_CSV)
    diameter_um = 2.0 * pores["pore_radius_m"].to_numpy(float) * 1e6
    volume_um3 = pores["pore_volume_m3"].to_numpy(float) * 1e18
    valid = np.isfinite(diameter_um) & np.isfinite(volume_um3) & (diameter_um > 0) & (volume_um3 > 0)
    diameter_um = diameter_um[valid]
    volume_um3 = volume_um3[valid]

    edges = np.linspace(diameter_um.min(), diameter_um.max(), 81)
    histogram, _ = np.histogram(diameter_um, bins=edges, weights=volume_um3)
    histogram = histogram / histogram.max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    pore_peak_um = float(centers[np.argmax(histogram)])
    t2_peak_ms = float(spectrum.loc[spectrum["amplitude"].idxmax(), "t2_ms"])

    t2_min, t2_max = 0.0, 2000.0
    pore_min = 0.0
    peak_position = (t2_peak_ms - t2_min) / (t2_max - t2_min)
    pore_max = pore_peak_um / peak_position

    pd.DataFrame(
        {
            "pore_diameter_left_um": edges[:-1],
            "pore_diameter_right_um": edges[1:],
            "pore_diameter_center_um": centers,
            "normalized_volume_weighted_frequency": histogram,
        }
    ).to_csv(LCURVE_DIR / "pore_diameter_linear_volume_weighted_histogram.csv", index=False)

    manifest = {
        "t2_spectrum": str((LCURVE_DIR / "lcurve_optimal_t2_spectrum.csv").resolve()),
        "pore_table": str(PORES_CSV.resolve()),
        "pore_count": int(valid.sum()),
        "weighting": "pore_volume_m3",
        "binning": "80 linear bins",
        "t2_peak_ms": t2_peak_ms,
        "pore_diameter_distribution_peak_um": pore_peak_um,
        "bottom_t2_axis_ms": [t2_min, t2_max],
        "top_pore_diameter_axis_um": [pore_min, pore_max],
        "alignment": "display-axis alignment only; T2 and pore diameter values are unchanged",
    }
    (LCURVE_DIR / "t2_pore_linear_overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        }
    )
    fig, ax_t2 = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax_pore = ax_t2.twiny()
    ax_t2.set_xlim(t2_min, t2_max)
    ax_pore.set_xlim(pore_min, pore_max)

    pore_line = ax_pore.stairs(
        histogram, edges, color="#657a9e", lw=2.0, label="volume-weighted pore diameter"
    )
    t2_line, = ax_t2.plot(
        spectrum["t2_ms"], spectrum["normalized_amplitude"],
        color="#1f2937", lw=2.4, label=r"L-curve T2 ($\alpha=39.19$)",
    )
    ax_t2.axvline(t2_peak_ms, color="#c54f3a", lw=1.1, ls="--", alpha=0.9)
    ax_t2.scatter([t2_peak_ms], [1.0], color="#c54f3a", s=28, zorder=4)
    ax_t2.set_xlabel(r"$T_2$ (ms)")
    ax_pore.set_xlabel("pore diameter (um)")
    ax_t2.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax_t2.set_ylim(-0.02, 1.08)
    ax_t2.grid(alpha=0.18)
    ax_t2.legend([pore_line, t2_line], [pore_line.get_label(), t2_line.get_label()], loc="upper left")
    ax_t2.set_title("Berea mesh-96: L-curve T2 vs pore-size distribution", pad=12, fontweight="bold")
    ax_t2.text(
        0.99, 0.03,
        f"aligned peaks: {t2_peak_ms:.1f} ms / {pore_peak_um:.2f} um\nlinear axes; values unchanged",
        transform=ax_t2.transAxes, ha="right", va="bottom", fontsize=8, color="0.3",
    )
    fig.savefig(LCURVE_DIR / "lcurve_t2_vs_pore_diameter_linear_aligned.png", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
