"""Nature-style figures for the ideal 2-D two-pore NMR/SIP study (PNG only, legend is the only annotation).

    python advanced_tools/ideal_two_pore_nature_plots.py overlay   # joint overlay with smoothing factors in the legend
    python advanced_tools/ideal_two_pore_nature_plots.py sweep     # throat width / length sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SKILL_SCRIPTS = Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

import advanced_tools.ideal_two_pore_channel_nmr_sip as base  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

BASE_OUT = ROOT / "simulation_outputs" / "ideal_two_pore_channel_nmr_sip_20261012"
SWEEP_OUT = ROOT / "simulation_outputs" / "ideal_two_pore_throat_sweep_nmr_20261012"
# Sweep inversion source; overridable from the command line (e.g. the MATLAB BRD re-inversion).
SWEEP_RAW = "sweep_raw.json"
SWEEP_METHOD = "L-curve NNLS"
SWEEP_TAG = ""
SIP_OUT = ROOT.parent / "SIP模拟" / "sip模拟" / "results" / "ideal_two_pore_channel_sip2d_20261012"

# Restrained palette: neutral bars, one NMR signal family, one SIP accent.
C_BODY, C_THROAT = "#9fb7cf", "#e3a782"
C_NMR, C_NMR_LIGHT, C_REF = "#2f6f8f", "#7fb3c9", "#5b5b5b"
C_SIP, C_SIP_ALT = "#c0643b", "#8a6bb0"
INK = "#1a1a1a"
MM = 1 / 25.4


def style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
    })


def plain_log_ticks(fig):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for ax in fig.findobj(matplotlib.axes.Axes):
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if "log" in scale or "function" in scale:
                axis.set_major_formatter(fmt)
                axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def export(fig, png: Path, multi_panel: bool, exclude_axes=()):
    """Alignment gate -> PNG -> throwaway QA PDF for text-size and collision audits.

    ``exclude_axes``: free-positioned insets, recorded as alignment exemptions.
    """

    plain_log_ticks(fig)
    fig.canvas.draw()
    if multi_panel:
        require_matplotlib_panel_alignment(fig, json_out=png.with_suffix(".alignment.json"), overlay_svg=None,
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exclude_axes=exclude_axes)
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    with tempfile.TemporaryDirectory() as tmp:
        qa = Path(tmp) / "figure_qa.pdf"
        fig.savefig(qa, bbox_inches="tight")
        for cmd in (
            [sys.executable, str(SKILL_SCRIPTS / "audit_pdf_text.py"), "--min-pt", "5", str(qa)],
            [sys.executable, str(SKILL_SCRIPTS / "audit_figure_collisions.py"), str(qa), "--json-out", str(png.with_suffix(".collision-audit.json"))],
        ):
            env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
            lines = [l for l in result.stdout.splitlines() if any(k in l for k in ("PASS", "FAIL", "WARN", "REVIEW", "NOT AUDITABLE", "min"))]
            print(png.name, Path(cmd[1]).name, "exit", result.returncode, "|", " / ".join(lines[:6]) or result.stderr.strip()[-300:])
    plt.close(fig)


def conversion_axes(ax):
    """Top axes: exact conversions of the radius axis (not a second data scale)."""

    t2 = ax.secondary_xaxis("top", functions=(
        lambda r: base.nmr_t2_from_radius_ms(np.clip(r, 1e-9, None)),
        lambda t: base.nmr_radius_um(np.clip(t, 1e-9, base.BULK_T2_MS * 0.999999))))
    t2.set_xlabel("NMR T2 (ms)")
    f = ax.secondary_xaxis(1.25, functions=(
        lambda r: base.sip_frequency_from_radius_hz(np.clip(r, 1e-9, None)),
        lambda q: base.sip_radius_from_frequency_um(np.clip(q, 1e-30, None))))
    f.set_xlabel("SIP relaxation frequency (Hz)")
    for secondary in (t2, f):
        secondary.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        secondary.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


XLIM = (5e-3, 60)


def clip(x, *ys, lo=XLIM[0], hi=XLIM[1]):
    """Keep only points inside the plotted radius range (no paths under tick labels)."""

    x = np.asarray(x, dtype=float)
    keep = (x >= lo) & (x <= hi)
    return (x[keep], *(np.asarray(y, dtype=float)[keep] for y in ys))


def nmr_density(t2, amplitude, lo=XLIM[0], hi=XLIM[1]):
    r, dens = base.density_in_log_radius(np.asarray(t2), np.asarray(amplitude))
    return clip(r, dens, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Question 4: joint overlay with the actual smoothing factors in the legend
# ---------------------------------------------------------------------------
def overlay():
    manifest = json.loads((BASE_OUT / "run_manifest.json").read_text(encoding="utf-8"))
    alpha = manifest["inversion"]["selected_alpha"]
    fixed_alpha = manifest["inversion"]["fixed_nnls_alpha"]
    sip_reg = manifest["sip"]["parameters"]
    spectra = pd.read_csv(BASE_OUT / "source_data" / "nmr_t2_spectra.csv")
    hist = pd.read_csv(BASE_OUT / "source_data" / "actual_pore_body_throat_size_histogram.csv")
    dd = pd.read_csv(SIP_OUT / "source_data" / "debye_decomposition_all_regional.csv")
    dd_uni = pd.read_csv(SIP_OUT / "source_data" / "debye_decomposition_all_uniform_niu.csv")
    sweep = pd.read_csv(SIP_OUT / "simulation_sweeps" / "all_regional" / "sweep_results.csv")
    edges = np.r_[hist["bin_left_um"], hist["bin_right_um"].iloc[-1]]
    widths = np.diff(edges)
    body, throat = hist["pore_body_area_fraction"].to_numpy(), hist["throat_area_fraction"].to_numpy()
    body_pct = ", ".join(f"{100 * v:.1f}%" for v in sorted(body[body > 0], reverse=True))
    throat_pct = f"{100 * throat.sum():.2f}%"

    fig, ax = plt.subplots(figsize=(150 * MM, 82 * MM))
    ax.bar(edges[:-1], body / body.max(), width=widths, align="edge", color=C_BODY, edgecolor="white", lw=0.6)
    ax.bar(edges[:-1], throat / throat.max(), width=widths, align="edge", color=C_THROAT, edgecolor="white", lw=0.6)
    r, d = nmr_density(spectra["t2_ms"], spectra["connected_fine_lcurve"])
    ax.fill_between(r, d / d.max(), color=C_NMR_LIGHT, alpha=0.35, lw=0)
    ax.plot(r, d / d.max(), color=C_NMR, lw=1.5)
    r, d = nmr_density(spectra["t2_ms"], spectra["uncoupled_lcurve"])
    ax.plot(r, d / d.max(), color=C_REF, lw=0.9, ls=":")
    ax.plot(*clip(dd["radius_um"], dd["amplitude_s_m"] / dd["amplitude_s_m"].max()), color=C_SIP, lw=1.5)
    ax.plot(*clip(base.sip_radius_from_frequency_um(sweep["frequency_hz"]), sweep["sigma_imag_s_m"] / sweep["sigma_imag_s_m"].max()), color=C_SIP, lw=0.9, ls="--")
    ax.plot(*clip(dd_uni["radius_um"], dd_uni["amplitude_s_m"] / dd_uni["amplitude_s_m"].max()), color=C_SIP_ALT, lw=0.9, ls="-.")
    ax.set(xscale="log", xlim=(5e-3, 60), ylim=(0, 1.05), xlabel="Pore radius r (µm)", ylabel="Amplitude / series maximum")
    conversion_axes(ax)
    handles = [
        Patch(color=C_BODY, label=f"Actual pore bodies, area-weighted ({body_pct} of pore area)"),
        Patch(color=C_THROAT, label=f"Actual throat, area-weighted ({throat_pct} of pore area)"),
        Line2D([], [], color=C_NMR, lw=1.5, label=f"NMR T2, connected, L-curve NNLS (α = {alpha['connected_fine']:.3g})"),
        Line2D([], [], color=C_REF, lw=0.9, ls=":", label=f"NMR T2, uncoupled reference, L-curve NNLS (α = {alpha['uncoupled']:.3g})"),
        Line2D([], [], color=C_SIP, lw=1.5, label="SIP Debye decomposition, region-resolved"),
        Line2D([], [], color=C_SIP, lw=0.9, ls="--", label="SIP σ''(f) at r = √(D+/(πf)), region-resolved"),
        Line2D([], [], color=C_SIP_ALT, lw=0.9, ls="-.", label="SIP Debye decomposition, Niu uniform upscaling"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.2, handlelength=2.2)
    out = BASE_OUT / "figures_nature"
    out.mkdir(exist_ok=True)
    export(fig, out / "overlay_actual_sizes_nmr_t2_sip.png", multi_panel=False)

    # Stacked absolute version (one y-scale per panel).
    fig, axes = plt.subplots(3, 1, figsize=(150 * MM, 150 * MM), sharex=True, gridspec_kw={"hspace": 0.12})
    a = axes[0]
    # Only non-empty bins: a zero-height bar on a log axis is drawn to -inf.
    for values, color, label in ((body, C_BODY, "Pore bodies"), (throat, C_THROAT, "Throats")):
        keep = values > 0
        a.bar(edges[:-1][keep], 100 * values[keep] - 0.05, bottom=0.05, width=widths[keep], align="edge", color=color, edgecolor="white", lw=0.6, label=label)
    a.set(yscale="log", ylim=(0.05, 300), ylabel="Pore-area fraction (%)")
    a.legend(loc="upper left")
    a = axes[1]
    for col, color, ls, lw, label in (
        ("connected_fine_lcurve", C_NMR, "-", 1.5, f"Connected, L-curve NNLS (α = {alpha['connected_fine']:.3g})"),
        (f"connected_fine_fixed_alpha{fixed_alpha:g}", C_NMR_LIGHT, "--", 1.0, f"Connected, fixed NNLS (α = {fixed_alpha:g})"),
        ("uncoupled_lcurve", C_REF, ":", 1.0, f"Uncoupled reference, L-curve NNLS (α = {alpha['uncoupled']:.3g})"),
    ):
        r, d = nmr_density(spectra["t2_ms"], spectra[col])
        a.plot(r, d, color=color, ls=ls, lw=lw, label=label)
    a.set(ylabel="NMR amplitude per log10 r")
    a.legend(loc="upper left")
    a = axes[2]
    a.plot(*clip(dd["radius_um"], dd["amplitude_s_m"] / dd["amplitude_s_m"].max()), color=C_SIP, lw=1.5, label="Debye decomposition, region-resolved")
    a.plot(*clip(base.sip_radius_from_frequency_um(sweep["frequency_hz"]), sweep["sigma_imag_s_m"] / sweep["sigma_imag_s_m"].max()), color=C_SIP, lw=0.9, ls="--", label="σ'', region-resolved")
    a.plot(*clip(dd_uni["radius_um"], dd_uni["amplitude_s_m"] / dd_uni["amplitude_s_m"].max()), color=C_SIP_ALT, lw=1.0, ls="-.", label="Debye decomposition, Niu uniform upscaling")
    a.set(xscale="log", xlim=(5e-3, 60), ylim=(0, 1.05), xlabel="Pore radius r (µm)", ylabel="SIP amplitude / maximum")
    a.legend(loc="upper center")
    conversion_axes(axes[0])  # after the shared x scale is final; a later scale change resets secondary formatters
    export(fig, out / "stacked_actual_nmr_sip_absolute.png", multi_panel=True)


# ---------------------------------------------------------------------------
# Question 5: throat width / length sweep
# ---------------------------------------------------------------------------
def sweep_tables():
    from advanced_tools.ideal_two_pore_throat_sweep_nmr import peak_metrics

    raw = json.loads((SWEEP_OUT / SWEEP_RAW).read_text(encoding="utf-8"))
    rows = []
    rho, dif, t2b = base.RHO2_UM_MS, base.DIFFUSION_UM2_MS, base.BULK_T2_MS
    rb, rs = base.GEOMETRY_UM["big_radius"], base.GEOMETRY_UM["small_radius"]
    t2s_iso, t2b_iso = base.bessel_disk_t2_ms(rs), base.bessel_disk_t2_ms(rb)
    rate_gap = 1 / t2s_iso - 1 / t2b_iso
    for c in raw:
        m = peak_metrics(c["t2_bins_ms"], c["spectrum"])
        w, length = c["width_um"], c["length_um"]
        # Diffusive exchange rate out of S through one throat, relative to the relaxation-rate gap.
        exchange = dif * w / (length * c["area_small_um2"]) / rate_gap
        rows.append({
            "width_um": w, "length_um": length, "aspect_L_over_w": length / w,
            "exchange_number": exchange,
            "throat_area_fraction": c["area_throat_um2"] / (c["area_big_um2"] + c["area_throat_um2"] + c["area_small_um2"]),
            "t2_small_region_fit_ms": c["t2_small_region_fit_ms"], "t2_big_region_fit_ms": c["t2_big_region_fit_ms"],
            "r_small_region_fit_um": float(base.nmr_radius_um(c["t2_small_region_fit_ms"])),
            "alpha": c["alpha"], "n_peaks": m["n_peaks"],
            "small_peak_t2_ms": m["small_peak_t2_ms"], "small_peak_fraction": m["small_peak_fraction"],
            "big_peak_t2_ms": m["big_peak_t2_ms"], "valley_to_lower_peak": m["valley_to_lower_peak"],
            "r_small_peak_um": float(base.nmr_radius_um(m["small_peak_t2_ms"])) if np.isfinite(m["small_peak_t2_ms"]) else np.nan,
            "r_big_peak_um": float(base.nmr_radius_um(m["big_peak_t2_ms"])),
        })
    table = pd.DataFrame(rows).sort_values(["width_um", "length_um"]).reset_index(drop=True)
    table.to_csv(SWEEP_OUT / f"sweep_summary{SWEEP_TAG}.csv", index=False)
    return raw, table


C_RESOLVED, C_MERGED = "#3a9a5b", "#8a4fa8"


PEAK_MIN_FRACTION = 0.02  # a T2 peak counts when its area exceeds 2% of the total amplitude (and T2 > 5 ms)
THROAT_ZOOM_BELOW = 0.02  # throat bars below 2% of the pore area get a magnified inset
GRID_XLIM = (0.0316, 39.8)


def zoom_inset(ax, lo, hi, y_top, draw, x_ticks, y_ticks, x_box_frac):
    """Dashed box [lo, hi] x [0, y_top] with an arrow to an inset that redraws only that window.

    The inset sits in the half of the panel away from the box; its tick labels are on the side
    away from the arrow so the arrow only touches a corner.
    """

    from matplotlib.patches import Rectangle

    box_height = max(y_top, 0.03)  # keep the dashed box visible when the zoomed quantity is tiny
    ax.add_patch(Rectangle((lo, 0), hi - lo, box_height, fill=False, ls="--", lw=0.6, edgecolor=INK, zorder=5))
    box_left_of_inset = x_box_frac < 0.35
    inset = ax.inset_axes([0.34 if box_left_of_inset else 0.24, 0.40, 0.24, 0.40])
    draw(inset)
    inset.set(xscale="log", xlim=(lo, hi), ylim=(0, y_top))
    inset.set_xticks(x_ticks)
    kept = []
    for v in sorted(set(y_ticks), reverse=True):  # drop key values closer than 20% of the inset height
        if all(abs(v - k) > 0.2 * y_top for k in kept):
            kept.append(v)
    y_ticks = sorted(kept)
    inset.set_yticks(y_ticks, [f"{v:.2g}" if v else "0" for v in y_ticks])
    inset.xaxis.tick_top()
    if box_left_of_inset:
        inset.yaxis.tick_right()
    inset.tick_params(labelsize=5.5, length=2, pad=1)
    for side in ("top", "right", "left", "bottom"):
        inset.spines[side].set_visible(True)
        inset.spines[side].set_linewidth(0.6)
    corner = (0.0, 0.0) if box_left_of_inset else (1.0, 0.0)
    from matplotlib.patches import ConnectionPatch

    arrow = ConnectionPatch(xyA=(np.sqrt(lo * hi), box_height), coordsA=ax.transData, xyB=corner, coordsB=inset.transAxes,
                            arrowstyle="->", lw=0.6, color=INK, shrinkA=0, shrinkB=0)
    ax.add_artist(arrow)
    return inset


def axis_fraction(r):
    return (np.log10(r) - np.log10(GRID_XLIM[0])) / (np.log10(GRID_XLIM[1]) - np.log10(GRID_XLIM[0]))


def sweep():
    raw, table = sweep_tables()
    widths = sorted(table["width_um"].unique())
    lengths = sorted(table["length_um"].unique())
    rb, rs = base.GEOMETRY_UM["big_radius"], base.GEOMETRY_UM["small_radius"]
    out = SWEEP_OUT / "figures_nature"
    out.mkdir(exist_ok=True)
    edges = 10.0 ** (np.arange(-20.5, 21.0, 1.0) / 10.0)
    bin_w = np.diff(edges)
    in_view = (edges[:-1] >= GRID_XLIM[0] * 0.999) & (edges[1:] <= GRID_XLIM[1] * 1.001)

    # Fig. 1: small multiples, rows = throat width, columns = throat length. One y scale per panel:
    # area fraction of the fixed total pore area (bodies and throat together) and the NMR amplitude
    # fraction re-binned into the same radius bins, so both sum to 1.
    fig, axes = plt.subplots(len(widths), len(lengths), figsize=(180 * MM, 200 * MM), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    lookup = {(c["width_um"], c["length_um"]): c for c in raw}
    insets = []
    peak_inset_drawn = False
    # Shared y limit: a T2 peak narrower than one histogram bin can exceed 1 on the per-bin scale.
    y_max = 1.0
    for c in raw:
        amp = np.asarray(c["spectrum"])
        _, dens = base.density_in_log_radius(np.asarray(c["t2_bins_ms"]), amp / amp.sum())
        y_max = max(y_max, float(dens.max()) * 0.1)
    y_lim = (0, 1.05 * y_max)
    for i, w in enumerate(widths):
        for j, length in enumerate(lengths):
            ax, c = axes[i, j], lookup[(w, length)]
            h = w / 2
            total = c["area_big_um2"] + c["area_small_um2"] + c["area_throat_um2"]
            body = np.zeros(edges.size - 1)
            throat = np.zeros(edges.size - 1)
            for radius, area, target in ((rb, c["area_big_um2"], body), (rs, c["area_small_um2"], body), (h, c["area_throat_um2"], throat)):
                target[np.searchsorted(edges, radius) - 1] += area / total
            t2 = np.asarray(c["t2_bins_ms"])
            amp = np.asarray(c["spectrum"])
            r_nmr = base.nmr_radius_um(t2)
            ok = np.isfinite(r_nmr)
            # T2 spectrum as a line on the same fraction axis: amplitude density per log10 r times the
            # histogram bin width (0.1 decade) = amplitude fraction expected in one histogram bin.
            r_line, dens_line = base.density_in_log_radius(t2, amp / amp.sum())
            y_line = dens_line * 0.1

            def line_max(window):
                inside = (r_line >= window[0]) & (r_line <= window[1])
                return float(y_line[inside].max()) if inside.any() else 0.0

            def draw(target_ax, lw_bar=0.4, lw_nmr=1.1, window=GRID_XLIM, cap=np.inf):
                # Only bins inside the axis window: paths outside a log axis are clamped into stray strokes.
                show = (edges[:-1] >= window[0] * 0.999) & (edges[1:] <= window[1] * 1.001)
                for values, color in ((body, C_BODY), (throat, C_THROAT)):
                    keep = show & (values > 0)  # empty bins would leave zero-height paths under tick labels
                    # cap: bars taller than an inset are cut at its top (no clipped path crossing its labels)
                    target_ax.bar(edges[:-1][keep], np.minimum(values[keep], cap), width=bin_w[keep], align="edge", color=color, edgecolor="white", lw=lw_bar)
                rr, yy = clip(r_line, y_line, lo=window[0], hi=window[1])
                target_ax.fill_between(rr, np.minimum(yy, cap), color=C_NMR_LIGHT, alpha=0.35, lw=0)
                target_ax.plot(rr, np.minimum(yy, cap), color=C_NMR, lw=lw_nmr)

            draw(ax)
            ax.set(xscale="log", xlim=GRID_XLIM, ylim=y_lim)
            ax.tick_params(labelsize=6)
            row = table[(table["width_um"] == w) & (table["length_um"] == length)].iloc[0]
            resolved = row["n_peaks"] >= 2
            for side in ("left", "right", "top", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(C_RESOLVED if resolved else C_MERGED)
                ax.spines[side].set_linewidth(1.2)
            throat_fraction = throat.sum()
            if throat_fraction < THROAT_ZOOM_BELOW:
                k = int(np.searchsorted(edges, h) - 1)
                lo, hi = edges[k] / 1.25, edges[k + 1] * 1.25
                y_top = 1.3 * max(throat_fraction, line_max((lo, hi)))
                insets.append(zoom_inset(ax, lo, hi, y_top, lambda a: draw(a, 0.4, 1.0, (lo, hi), y_top), [h], [0, float(f"{throat_fraction:.2g}")], axis_fraction(h)))
            if (i, j) == (len(widths) - 1, len(lengths) - 1) and resolved:
                peak_inset_drawn = True
                r_peak = row["r_small_peak_um"]
                lo, hi = r_peak / 1.6, rs * 1.6
                window = (edges[:-1] >= lo) & (edges[1:] <= hi)
                # Height set by the small-pore peak and the S bar; a taller throat bar is clipped at the top.
                y_top = 1.3 * max(line_max((lo, hi)), body[window].max())
                insets.append(zoom_inset(ax, lo, hi, y_top, lambda a: draw(a, 0.4, 1.0, (lo, hi), y_top), [float(f"{r_peak:.2g}"), rs],
                                         [0, float(f"{line_max((lo, hi)):.2g}"), float(f"{body[window].max():.2g}")], axis_fraction(np.sqrt(lo * hi))))
            if j == 0:
                ax.set_ylabel(f"w = {w:g} µm\nFraction", fontsize=6.5)
            if i == len(widths) - 1:
                ax.set_xlabel(f"r (µm)\nL = {length:g} µm", fontsize=6.5)
    handles = [
        Patch(color=C_BODY, label="Actual pore bodies (B, S): fraction of total pore area"),
        Patch(color=C_THROAT, label="Actual throat: fraction of total pore area"),
        Line2D([], [], color=C_NMR, lw=1.1, label=f"NMR T2 spectrum ({SWEEP_METHOD}), amplitude fraction per 0.1-decade bin; r = 2ρ2/(1/T2 − 1/T2B)"),
        Patch(facecolor="white", edgecolor=C_RESOLVED, lw=1.2, label=f"Two T2 peaks resolved: second peak > {PEAK_MIN_FRACTION:.0%} of total T2 amplitude, T2 > 5 ms"),
        Patch(facecolor="white", edgecolor=C_MERGED, lw=1.2, label="Single T2 peak (no second peak above the threshold)"),
        Patch(facecolor="none", edgecolor=INK, ls="--", lw=0.6, label=f"Dashed box → inset: throat area < {THROAT_ZOOM_BELOW:.0%}" + ("; small-pore peak (w = 4, L = 30 µm)" if peak_inset_drawn else "")),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2, columnspacing=1.2)
    export(fig, out / f"sweep_overlay_grid{SWEEP_TAG}.png", multi_panel=True, exclude_axes=insets)

    # Fig. 2: what controls peak merging.
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM), gridspec_kw={"wspace": 0.62})
    # Categorical map (the merge indicator only takes the values 0 and 1), same colours as the grid frames.
    resolved_grid = (table.pivot(index="width_um", columns="length_um", values="n_peaks").loc[widths, lengths] >= 2).to_numpy()
    a = axes[0]
    a.pcolormesh(np.arange(len(lengths) + 1), np.arange(len(widths) + 1), resolved_grid.astype(float),
                 cmap=matplotlib.colors.ListedColormap([C_MERGED, C_RESOLVED]), vmin=0, vmax=1, edgecolors="white", lw=0.6)
    a.set_xticks(np.arange(len(lengths)) + 0.5, [f"{v:g}" for v in lengths])
    a.set_yticks(np.arange(len(widths)) + 0.5, [f"{v:g}" for v in widths])
    a.set(xlabel="Throat length L (µm)", ylabel="Throat width w (µm)")
    markers = dict(zip(lengths, ["o", "s", "^", "D"]))
    colors = dict(zip(widths, plt.cm.viridis(np.linspace(0.1, 0.85, len(widths)))))
    a = axes[1]
    for _, row in table.iterrows():
        a.scatter(row["exchange_number"], row["r_small_region_fit_um"] / rs, marker=markers[row["length_um"]], color=colors[row["width_um"]], s=16, edgecolor=INK, lw=0.3)
    a.axhline(1.0, color=C_REF, lw=0.7, ls=":")
    a.set(xscale="log", xlabel="Exchange number Ψ", ylabel="r_NMR(S) / R_S")
    a = axes[2]
    for _, row in table.iterrows():
        a.scatter(row["exchange_number"], row["t2_small_region_fit_ms"], marker=markers[row["length_um"]], color=colors[row["width_um"]], s=16, edgecolor=INK, lw=0.3)
        a.scatter(row["exchange_number"], row["t2_big_region_fit_ms"], marker=markers[row["length_um"]], facecolor="white", edgecolor=colors[row["width_um"]], s=16, lw=0.8)
    a.axhline(base.bessel_disk_t2_ms(rs), color=C_REF, lw=0.7, ls=":")
    a.axhline(base.bessel_disk_t2_ms(rb), color=C_REF, lw=0.7, ls="--")
    a.set(xscale="log", yscale="log", xlabel="Exchange number Ψ", ylabel="Region T2, direct fit (ms)")
    # Every legend symbol is drawn exactly as a plotted point: colour entries use the circle (L = 2 µm),
    # shape entries use the colour of w = 0.1 µm, and filled/open entries are that same point (S / B).
    ref_color, ref_marker = colors[widths[0]], markers[lengths[0]]
    legend = [Patch(facecolor=C_RESOLVED, edgecolor="white", label=f"Map: two T2 peaks (second > {PEAK_MIN_FRACTION:.0%})"),
              Patch(facecolor=C_MERGED, edgecolor="white", label="Map: one T2 peak")]
    legend += [Line2D([], [], ls="", marker=ref_marker, color=colors[w], markeredgecolor=INK, markeredgewidth=0.3, ms=4,
                      label=f"Colour: w = {w:g} µm") for w in widths]
    legend += [Line2D([], [], ls="", marker=markers[L], color=ref_color, markeredgecolor=INK, markeredgewidth=0.3, ms=4,
                      label=f"Shape: L = {L:g} µm") for L in lengths]
    legend += [Line2D([], [], ls="", marker=ref_marker, color=ref_color, markeredgecolor=INK, markeredgewidth=0.3, ms=4, label="Filled: small pore S"),
               Line2D([], [], ls="", marker=ref_marker, markerfacecolor="white", markeredgecolor=ref_color, markeredgewidth=0.8, ms=4, label="Open: big pore B (right panel)"),
               Line2D([], [], color=C_REF, lw=0.7, ls=":", label="Isolated S (analytic); also r/R_S = 1"),
               Line2D([], [], color=C_REF, lw=0.7, ls="--", label="Isolated B (analytic)"),
               Line2D([], [], ls="", label="Ψ = D·w / (L·A_S) ÷ (1/T2,S − 1/T2,B), isolated-pore rates")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, columnspacing=1.0)
    export(fig, out / f"sweep_coupling_controls{SWEEP_TAG}.png", multi_panel=True)


def geometry_grid():
    """Companion of sweep_overlay_grid.png: the simulated closed geometries, same rows/columns, true scale."""

    from matplotlib.patches import Circle, Rectangle

    widths = sweep_module().WIDTHS_UM
    lengths = sweep_module().LENGTHS_UM
    rb, rs = base.GEOMETRY_UM["big_radius"], base.GEOMETRY_UM["small_radius"]
    x_max = rb + max(lengths) + 2 * rs  # right edge of S for the longest throat
    out = SWEEP_OUT / "figures_nature"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(widths), len(lengths), figsize=(180 * MM, 128 * MM), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    for i, w in enumerate(widths):
        for j, length in enumerate(lengths):
            ax = axes[i, j]
            xs = rb + length + rs  # S centre; B centre at the origin
            h = 0.5 * w
            # Throat drawn wall to wall, overlapping the disks so the mouths stay closed at any width.
            x0, x1 = np.sqrt(rb**2 - h**2) - 0.5, xs - np.sqrt(rs**2 - h**2) + 0.5
            ax.add_patch(Rectangle((x0, -h), x1 - x0, w, facecolor=C_THROAT, edgecolor=C_THROAT, lw=0.5, zorder=1))
            ax.add_patch(Circle((0, 0), rb, facecolor=C_BODY, edgecolor="#5f7f9f", lw=0.6, zorder=2))
            ax.add_patch(Circle((xs, 0), rs, facecolor=C_BODY, edgecolor="#5f7f9f", lw=0.6, zorder=2))
            ax.set(xlim=(-rb - 1, x_max + 1), ylim=(-rb - 1, rb + 1), aspect="equal")
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f"w = {w:g} µm\ny (µm)", fontsize=6.5)
            if i == len(widths) - 1:
                ax.set_xlabel(f"x (µm)\nL = {length:g} µm", fontsize=6.5)
    handles = [
        Patch(facecolor=C_BODY, edgecolor="#5f7f9f", label=f"Pore bodies: B (R = {rb:g} µm), S (R = {rs:g} µm)"),
        Patch(facecolor=C_THROAT, edgecolor=C_THROAT, label="Throat (width w, wall-to-wall length L); 0.5 pt outline keeps w < 0.5 µm visible"),
        Patch(facecolor="white", edgecolor="#bbbbbb", label="Solid; all walls relax with ρ2 = 5 µm/s; drawn to a common true scale"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=1)
    export(fig, out / "sweep_geometry_grid.png", multi_panel=True)


def sweep_module():
    import advanced_tools.ideal_two_pore_throat_sweep_nmr as module

    return module


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["overlay", "sweep", "geometry"])
    parser.add_argument("--sweep-dir", type=Path, default=None)
    parser.add_argument("--raw", default=None, help="sweep raw JSON inside --sweep-dir")
    parser.add_argument("--method", default=None, help="inversion label used in legends")
    parser.add_argument("--tag", default=None, help="suffix for output file names")
    args = parser.parse_args()
    global SWEEP_OUT, SWEEP_RAW, SWEEP_METHOD, SWEEP_TAG
    SWEEP_OUT = args.sweep_dir.resolve() if args.sweep_dir else SWEEP_OUT
    SWEEP_RAW = args.raw or SWEEP_RAW
    SWEEP_METHOD = args.method or SWEEP_METHOD
    SWEEP_TAG = args.tag or SWEEP_TAG
    style()
    {"overlay": overlay, "sweep": sweep, "geometry": geometry_grid}[args.figure]()


if __name__ == "__main__":
    main()
