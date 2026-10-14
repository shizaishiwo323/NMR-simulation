"""NMR-only overlay for the ideal 2-D pore chain: actual pore-size histogram + inverted T2 mapped to radius.

Reads source data written by ``ideal_two_pore_channel_nmr_sip.py``; exports a 600-dpi PNG only.

    python advanced_tools/plot_ideal_two_pore_channel_nmr_overlay.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "simulation_outputs" / "ideal_two_pore_channel_nmr_sip_20261012"
RHO2_UM_MS = 0.005  # surface relaxivity used in the simulation (5 um/s)
BULK_T2_MS = 3000.0

C_BODY, C_THROAT, C_NMR, C_REF = "#8fb0cc", "#e3a27a", "#2c6e8f", "#555555"
C_THROAT_LINE = "#b3541e"  # darker than the throat bar so the magnified curve stays visible on it


def t2_from_radius(r):
    return 1.0 / (2.0 * RHO2_UM_MS / np.clip(r, 1e-9, None) + 1.0 / BULK_T2_MS)


def radius_from_t2(t2):
    rate = 1.0 / np.clip(t2, 1e-9, BULK_T2_MS * 0.999999) - 1.0 / BULK_T2_MS
    return 2.0 * RHO2_UM_MS / rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--qa-scripts", type=Path, default=None, help="nature-figure scripts dir; runs PDF audits on a temporary file")
    args = parser.parse_args()
    src = args.run_dir / "source_data"
    hist = pd.read_csv(src / "actual_pore_body_throat_size_histogram.csv")
    series = pd.read_csv(src / "overlay_series_normalized.csv")
    spectra = pd.read_csv(src / "nmr_t2_spectra.csv")
    lcurve = pd.read_csv(src / "nmr_lcurve_connected_fine.csv")
    alpha = float(lcurve.loc[lcurve["selected"], "alpha"].iloc[0])

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.8,
        "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.5,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.55))

    left, right = hist["bin_left_um"].to_numpy(), hist["bin_right_um"].to_numpy()
    body, throat = hist["pore_body_area_fraction"].to_numpy(), hist["throat_area_fraction"].to_numpy()
    body_pct = ", ".join(f"{100 * v:.1f}%" for v in sorted(body[body > 0], reverse=True))
    ax.bar(left, body / body.max(), width=right - left, align="edge", color=C_BODY, lw=0,
           label=f"Pore bodies, area-weighted ({body_pct} of pore area)")
    ax.bar(left, throat / throat.max(), width=right - left, align="edge", color=C_THROAT, lw=0,
           label=f"Throat, area-weighted ({100 * throat.sum():.2f}% of pore area)")

    xlim = (0.02, 50.0)
    in_view = series["radius_um"].between(*xlim)
    nmr = series[(series["series"] == "nmr") & in_view]
    ax.fill_between(nmr["radius_um"], nmr["normalized_amplitude"], color=C_NMR, alpha=0.15, lw=0)
    ax.plot(nmr["radius_um"], nmr["normalized_amplitude"], color=C_NMR, lw=1.4,
            label=f"NMR T2, connected chain (coupled simulation; L-curve NNLS, α = {alpha:.3g})")

    # Isolated pores: each pore simulated alone and its own decay inverted alone, then weighted by its
    # pore-area fraction so relative amplitudes stay physical. Bodies use the disk map, the throat the slit map.
    t2 = spectra["t2_ms"].to_numpy()
    weight = {"big": body.max(), "small": np.sort(body[body > 0])[0], "throat": throat.sum()}
    curves = {}
    for name, factor in (("big", 2.0), ("small", 2.0), ("throat", 1.0)):
        amp = spectra[f"isolated_{name}_lcurve"].to_numpy()
        rate = 1.0 / t2 - 1.0 / BULK_T2_MS
        ok = rate > 0
        r = factor * RHO2_UM_MS / rate[ok]
        dens = weight[name] * amp[ok] / amp.sum() / np.gradient(np.log10(r))
        curves[name] = (r, dens)
    r_body = curves["big"][0]
    body_dens = curves["big"][1] + curves["small"][1]
    scale = body_dens.max()
    keep = (r_body >= xlim[0]) & (r_body <= xlim[1])
    ax.plot(r_body[keep], body_dens[keep] / scale, color=C_REF, lw=1.0, ls=(0, (1.2, 1.2)),
            label="NMR T2, isolated pore bodies (separate simulation + inversion)")
    r_th, th_dens = curves["throat"]
    peak = th_dens.max() / scale
    magnify = next(m for m in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if peak * m >= 0.3)
    keep = (r_th >= xlim[0]) & (r_th <= xlim[1])
    # True scale in the main panel (a sliver at the baseline); the magnified view lives only in the inset.
    ax.plot(r_th[keep], th_dens[keep] / scale, color=C_THROAT_LINE, lw=1.2, ls=(0, (1.2, 1.2)),
            label=f"NMR T2, isolated throat (separate; slit map h = ρ2/(1/T2 − 1/T2B)); inset ×{magnify}")

    # Throat zoom: box on the main panel, arrow to an inset that shows only the throat region.
    zoom_x, zoom_y = (0.036, 0.072), (0.0, 0.09)
    ax.add_patch(matplotlib.patches.Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0],
                                              fill=False, ec=C_THROAT_LINE, lw=0.8, zorder=5))
    inset_box = (0.30, 0.42, 0.30, 0.40)  # axes fraction: empty space between throat and pore bodies
    axins = ax.inset_axes(inset_box)
    th_bin = (hist["throat_area_fraction"] > 0).to_numpy()
    axins.bar(left[th_bin], np.ones(th_bin.sum()), width=(right - left)[th_bin], align="edge", color=C_THROAT, lw=0)
    sel = (r_th >= zoom_x[0]) & (r_th <= zoom_x[1])
    axins.plot(r_th[sel], magnify * th_dens[sel] / scale, color=C_THROAT_LINE, lw=1.2, ls=(0, (1.2, 1.2)))
    axins.set_xscale("log")
    axins.set_xlim(*zoom_x)
    axins.set_ylim(0, 1.06)
    axins.set_xticks([0.04, 0.05, 0.06, 0.07])
    axins.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axins.set_yticks([0, 0.5, 1.0])
    axins.tick_params(labelsize=5.5, length=2, pad=1.5)
    axins.set_ylabel(f"NMR ×{magnify}", fontsize=5.5, labelpad=1)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(C_THROAT_LINE)
        spine.set_linewidth(0.8)
    ax.annotate("", xy=(inset_box[0], inset_box[1]), xycoords="axes fraction",
                xytext=(zoom_x[1], zoom_y[1]), textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=C_THROAT_LINE, lw=0.8, shrinkA=0, shrinkB=2))

    ax.set_xscale("log")  # radii are strictly positive (filtered to xlim above)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Pore radius r (µm)")
    ax.set_ylabel("Amplitude / series maximum")
    top = ax.secondary_xaxis("top", functions=(t2_from_radius, radius_from_t2))
    top.set_xlabel("T2 (ms)")
    top.spines["top"].set_linewidth(0.8)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")  # avoids sub-5-pt superscript exponents
    for axis in (ax.xaxis, top.xaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1, fontsize=6.3, handlelength=1.8, borderaxespad=0)

    out = args.run_dir / "figures" / "fig7_overlay_actual_sizes_nmr_t2_only.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    if args.qa_scripts is not None:
        with tempfile.TemporaryDirectory() as tmp:
            qa_pdf = Path(tmp) / "figure_qa.pdf"
            fig.savefig(qa_pdf, bbox_inches="tight")
            for script in ("audit_pdf_text.py", "audit_figure_collisions.py"):
                subprocess.run([sys.executable, str(args.qa_scripts / script), str(qa_pdf)], check=False)
    print(out.name)


if __name__ == "__main__":
    sys.exit(main())
