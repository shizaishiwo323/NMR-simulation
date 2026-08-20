"""Plot L-curve T2 spectrum against volume-weighted throat diameters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path("simulation_outputs/niu2020_berea_porosity_grouped_nmr_mesh96_alpha0p1_quality0")
LCURVE_DIR = RUN_DIR / "lcurve_inversion"
THROATS_CSV = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\results\niu2020_berea_reproduction3"
    r"\pore_network\pnextract\network_parsed\throats.csv"
)


def main() -> None:
    spectrum = pd.read_csv(LCURVE_DIR / "lcurve_optimal_t2_spectrum.csv")
    throats = pd.read_csv(THROATS_CSV)
    diameter_um = 2.0 * throats["throat_radius_m"].to_numpy(float) * 1e6
    volume_um3 = throats["throat_volume_m3"].to_numpy(float) * 1e18
    internal = (
        np.isfinite(diameter_um)
        & np.isfinite(volume_um3)
        & (diameter_um > 0)
        & (volume_um3 > 0)
        & (throats["pore1_id"].to_numpy(int) >= 0)
        & (throats["pore2_id"].to_numpy(int) >= 0)
    )
    diameter_um = diameter_um[internal]
    volume_um3 = volume_um3[internal]

    edges = np.logspace(np.log10(diameter_um.min()), np.log10(diameter_um.max()), 81)
    histogram, _ = np.histogram(diameter_um, bins=edges, weights=volume_um3)
    histogram = histogram / histogram.max()
    centers = np.sqrt(edges[:-1] * edges[1:])
    throat_peak_um = float(centers[np.argmax(histogram)])
    t2_peak_ms = float(spectrum.loc[spectrum["amplitude"].idxmax(), "t2_ms"])

    t2_min, t2_max = 1e-2, 1e5
    throat_min = 2.0
    peak_position = (np.log10(t2_peak_ms) - np.log10(t2_min)) / (np.log10(t2_max) - np.log10(t2_min))
    throat_max = 10 ** (
        np.log10(throat_min) + (np.log10(throat_peak_um) - np.log10(throat_min)) / peak_position
    )

    hist_frame = pd.DataFrame(
        {
            "throat_diameter_left_um": edges[:-1],
            "throat_diameter_right_um": edges[1:],
            "throat_diameter_center_um": centers,
            "normalized_volume_weighted_frequency": histogram,
        }
    )
    hist_frame.to_csv(LCURVE_DIR / "internal_throat_volume_weighted_histogram.csv", index=False)
    manifest = {
        "t2_spectrum": str((LCURVE_DIR / "lcurve_optimal_t2_spectrum.csv").resolve()),
        "throat_table": str(THROATS_CSV.resolve()),
        "internal_throat_count": int(internal.sum()),
        "weighting": "throat_volume_m3",
        "t2_peak_ms": t2_peak_ms,
        "throat_distribution_peak_um": throat_peak_um,
        "bottom_t2_axis_ms": [t2_min, t2_max],
        "top_throat_axis_um": [throat_min, throat_max],
        "alignment": "display-axis alignment only; T2 and throat values are unchanged",
    }
    (LCURVE_DIR / "t2_throat_overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
    ax_throat = ax_t2.twiny()
    ax_t2.set_xscale("log")
    ax_throat.set_xscale("log")
    ax_t2.set_xlim(t2_min, t2_max)
    ax_throat.set_xlim(throat_min, throat_max)

    throat_line = ax_throat.stairs(
        histogram, edges, color="#657a9e", lw=2.0, label="volume-weighted internal throat diameter"
    )
    t2_line, = ax_t2.plot(
        spectrum["t2_ms"], spectrum["normalized_amplitude"],
        color="#1f2937", lw=2.4, label=r"L-curve T2 ($\alpha=39.19$)",
    )
    ax_t2.axvline(t2_peak_ms, color="#c54f3a", lw=1.1, ls="--", alpha=0.9)
    ax_t2.scatter([t2_peak_ms], [1.0], color="#c54f3a", s=28, zorder=4)

    ax_t2.set_xlabel(r"$T_2$ (ms)")
    ax_throat.set_xlabel("throat diameter (um)")
    ax_t2.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax_t2.set_ylim(-0.02, 1.08)
    ax_t2.grid(alpha=0.18, which="both")
    ax_t2.legend([throat_line, t2_line], [throat_line.get_label(), t2_line.get_label()], loc="upper left")
    ax_t2.set_title("Berea mesh-96: L-curve T2 vs pore-throat distribution", pad=12, fontweight="bold")
    ax_t2.text(
        0.99, 0.03,
        f"aligned peaks: {t2_peak_ms:.1f} ms / {throat_peak_um:.2f} um\ndisplay-axis alignment; values unchanged",
        transform=ax_t2.transAxes, ha="right", va="bottom", fontsize=8, color="0.3",
    )
    fig.savefig(LCURVE_DIR / "lcurve_t2_vs_throat_distribution_aligned.png", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
