"""Figures for the coupled two-pore NMR back-calculation analysis (Nature style: PNG only, legend-only text).

    python advanced_tools/plot_coupled_two_pore_backcalc.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import advanced_tools.coupled_two_pore_backcalc as cb  # noqa: E402
import advanced_tools.ideal_two_pore_nature_plots as nfp  # noqa: E402

OUT = cb.OUT
FIG = OUT / "figures"
MM = nfp.MM
INK, GREY = nfp.INK, "#6b6b6b"
REGIME_COLORS = {"I": "#3a7ca5", "II": "#d9894f", "III": "#7a4f9a"}
REGIME_LABELS = {"I": "Regime I, slow exchange (η > 0.8)", "II": "Regime II, intermediate (0.1 < η < 0.8)", "III": "Regime III, fast exchange (η < 0.1)"}
CASE_COLORS = ["#2f6f8f", "#5a9e6f", "#d9894f", "#a23b3b", "#7a4f9a"]


def case_label(w, length):
    return f"w = {w:g} µm, L = {length:g} µm"


def universal_curve(psi):
    a_s, _, r_s = cb.disk(cb.R_SMALL)
    a_b, _, r_b = cb.disk(cb.R_BIG)
    p_s = a_s / (a_s + a_b)
    rows = []
    for x in psi:
        k_s = x * (r_s - r_b)
        lf, ls, af = cb.two_site(r_s, r_b, k_s, k_s * p_s / (1 - p_s), p_s)
        rows.append((1 / lf, 1 / ls, af / p_s))
    return np.array(rows), p_s


def fig_regimes(table):
    widths = sorted(table.width_um.unique())
    lengths = sorted(table.length_um.unique())
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), gridspec_kw={"wspace": 0.42})
    a = axes[0]
    codes = {"I": 0, "II": 1, "III": 2}
    grid = table.pivot(index="width_um", columns="length_um", values="regime").loc[widths, lengths].map(codes.get).to_numpy(float)
    a.pcolormesh(np.arange(len(lengths) + 1), np.arange(len(widths) + 1), grid,
                 cmap=matplotlib.colors.ListedColormap([REGIME_COLORS[k] for k in ("I", "II", "III")]), vmin=-0.5, vmax=2.5,
                 edgecolors="white", lw=0.8)
    a.set_xticks(np.arange(len(lengths)) + 0.5, [f"{v:g}" for v in lengths])
    a.set_yticks(np.arange(len(widths)) + 0.5, [f"{v:g}" for v in widths])
    a.set(xlabel="Throat length L (µm)", ylabel="Throat width w (µm)")

    psi = np.logspace(-3, 2, 300)
    curve, p_s = universal_curve(psi)
    a = axes[1]
    a.plot(psi, curve[:, 2], color=INK, lw=1.1)
    a.axhline(0.8, color=GREY, lw=0.6, ls=":")
    a.axhline(0.1, color=GREY, lw=0.6, ls=":")
    for _, r in table.iterrows():
        a.scatter(r.Psi_eff, r.tl_amp_fast / r.p_small, s=18, color=REGIME_COLORS[r.regime], edgecolor=INK, lw=0.3, zorder=3)
        a.scatter(r.Psi_eff, r.nnls_amp_fast / r.p_small, s=18, facecolor="white", edgecolor=REGIME_COLORS[r.regime], lw=0.9, zorder=4)
    a.set(xscale="log", yscale="log", ylim=(5e-4, 1.6), xlim=(5e-4, 60), xlabel="Effective exchange number Ψ_eff",
          ylabel="Retained small-pore amplitude η = A_fast / p_S")
    a = axes[2]
    a.plot(psi, curve[:, 0], color=INK, lw=1.1)
    a.plot(psi, curve[:, 1], color=INK, lw=1.1, ls="--")
    a.axhline(cb.base.bessel_disk_t2_ms(cb.R_SMALL), color=GREY, lw=0.6, ls=":")
    a.axhline(cb.base.bessel_disk_t2_ms(cb.R_BIG), color=GREY, lw=0.6, ls="-.")
    for _, r in table.iterrows():
        c = REGIME_COLORS[r.regime]
        a.scatter(r.Psi_eff, r.tl_t2_fast_ms, s=18, color=c, edgecolor=INK, lw=0.3, zorder=3)
        a.scatter(r.Psi_eff, r.tl_t2_slow_ms, s=18, marker="s", color=c, edgecolor=INK, lw=0.3, zorder=3)
        if np.isfinite(r.nnls_t2_fast_lm_ms):
            a.scatter(r.Psi_eff, r.nnls_t2_fast_lm_ms, s=18, facecolor="white", edgecolor=c, lw=0.9, zorder=4)
    a.set(xscale="log", yscale="log", xlim=(5e-4, 60), ylim=(10, 2500), xlabel="Effective exchange number Ψ_eff", ylabel="Mode T2 (ms)")
    ref = REGIME_COLORS["II"]
    handles = [Patch(color=REGIME_COLORS[k], label=REGIME_LABELS[k]) for k in ("I", "II", "III")]
    handles += [
        Line2D([], [], color=INK, lw=1.1, label="Two-site model (no throat sink), plotted against Ψ: fast mode / η"),
        Line2D([], [], color=INK, lw=1.1, ls="--", label="Two-site model: slow mode"),
        Line2D([], [], ls="", marker="o", color=ref, markeredgecolor=INK, markeredgewidth=0.3, ms=4.5, label="1-D throat model (TL), fast mode"),
        Line2D([], [], ls="", marker="s", color=ref, markeredgecolor=INK, markeredgewidth=0.3, ms=4.5, label="1-D throat model (TL), slow mode"),
        Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=ref, markeredgewidth=0.9, ms=4.5, label="NNLS minor peak (L-curve, SNR 1000)"),
        Line2D([], [], color=GREY, lw=0.6, ls=":", label="η = 0.8 and 0.1 (regime limits); isolated S T2 (right)"),
        Line2D([], [], color=GREY, lw=0.6, ls="-.", label="Isolated B T2 (right)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, columnspacing=1.0)
    nfp.export(fig, FIG / "fig1_regimes_and_universal_curves.png", multi_panel=True)


def fig_validation(table, cases):
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 60 * MM), gridspec_kw={"wspace": 0.45})
    a = axes[0]
    for color, key in zip(CASE_COLORS, cb.REPRESENTATIVE):
        c = cases[key]
        row = table[(table.width_um == key[0]) & (table.length_um == key[1])].iloc[0]
        t = np.arange(len(c["decay_every_1ms"]), dtype=float)
        a.plot(t, c["decay_every_1ms"], color=color, lw=1.2)
        h = 0.5 * key[0]
        a_s, _, r_s = cb.disk(cb.R_SMALL, h)
        a_b, _, r_b = cb.disk(cb.R_BIG, h)
        modes = cb.tl_modes(a_s, a_b, r_s, r_b, key[0], row.throat_length_mesh_um)
        tm = t[::400]
        a.plot(tm, sum(amp * np.exp(-lam * tm) for lam, amp in modes), ls="", marker="o", ms=3, mfc="white", mec=color, mew=0.8)
    a.set(yscale="log", xlim=(0, 6000), ylim=(4e-3, 1.05), xlabel="Time (ms)", ylabel="M(t) / M(0)")
    a = axes[1]
    lim = (10, 600)
    a.plot(lim, lim, color=GREY, lw=0.6, ls=":")
    for _, r in table.iterrows():
        a.scatter(r.tl_t2_fast_ms, r.nnls_t2_fast_lm_ms, s=18, color=REGIME_COLORS[r.regime], edgecolor=INK, lw=0.3)
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="TL model fast-mode T2 (ms)", ylabel="NNLS minor-peak T2lm (ms)")
    a = axes[2]
    lim = (5e-5, 0.1)
    a.plot(lim, lim, color=GREY, lw=0.6, ls=":")
    for _, r in table.iterrows():
        a.scatter(r.tl_amp_fast, r.nnls_amp_fast, s=18, color=REGIME_COLORS[r.regime], edgecolor=INK, lw=0.3)
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="TL model fast-mode amplitude", ylabel="NNLS minor-peak fraction")
    handles = [Line2D([], [], color=col, lw=1.2, label=f"Simulation, {case_label(*k)}") for col, k in zip(CASE_COLORS, cb.REPRESENTATIVE)]
    handles += [Line2D([], [], ls="", marker="o", ms=3, mfc="white", mec=CASE_COLORS[0], mew=0.8, label="TL model (same colour), no fitted parameter")]
    handles += [Patch(color=REGIME_COLORS[k], label=REGIME_LABELS[k]) for k in ("I", "II", "III")]
    handles += [Line2D([], [], color=GREY, lw=0.6, ls=":", label="1:1")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, columnspacing=1.0)
    nfp.export(fig, FIG / "fig2_model_validation.png", multi_panel=True)


def fig_backcalc(back):
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), gridspec_kw={"wspace": 0.25})
    x = np.arange(len(back))
    routes = [("A_r", "o", "#5b5b5b", "A: isolated-pore formula per peak"),
              ("B_r", "^", "#d9894f", "B: amplitude ratio, r_S = r_B √(f_S/f_B)"),
              ("D_r", "s", "#2f6f8f", "D: TL model, throat w, L known (NNLS input)"),
              ("D_exactobs_r", "D", "#5a9e6f", "D with exact model modes (consistency check)")]
    offsets = np.linspace(-0.24, 0.24, len(routes))
    for a, size, truth in ((axes[0], "small", cb.R_SMALL), (axes[1], "big", cb.R_BIG)):
        a.axhline(truth, color=INK, lw=0.8)
        if size == "big":
            a.axhline(back.R_eff_true_um.iloc[0], color=GREY, lw=0.8, ls="--")
        for (prefix, marker, color, _), dx in zip(routes, offsets):
            vals = back[f"{prefix}_{size}"].to_numpy(float)
            a.scatter(x + dx, vals, marker=marker, s=22, color=color, edgecolor=INK, lw=0.3, zorder=3)
        a.set_xticks(x, [f"w = {w:g}\nL = {l:g}\n({rg})" for w, l, rg in zip(back.width_um, back.length_um, back.regime)])
        a.set_yscale("log" if size == "small" else "linear")
        a.set(xlim=(-0.6, len(back) - 0.4), ylabel=("Small-pore" if size == "small" else "Big-pore") + " radius (µm)",
              xlabel="Representative case: throat w, L (µm) and regime")
    axes[0].set_ylim(0.1, 30)
    axes[1].set_ylim(0, 25)
    handles = [Line2D([], [], ls="", marker=m, color=c, markeredgecolor=INK, markeredgewidth=0.3, ms=5, label=lab) for _, m, c, lab in routes]
    handles += [Line2D([], [], color=INK, lw=0.8, label="True radius (R_S = 5 µm, R_B = 20 µm)"),
                Line2D([], [], color=GREY, lw=0.8, ls="--", label="R_eff = 2(A_S + A_B)/(P_S + P_B) = 17.0 µm (right)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, columnspacing=1.0)
    nfp.export(fig, FIG / "fig3_backcalculated_sizes.png", multi_panel=True)


def fig_spectra(cases, snr, table):
    fig, axes = plt.subplots(1, len(cb.REPRESENTATIVE), figsize=(180 * MM, 52 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, key in zip(axes, cb.REPRESENTATIVE):
        c = cases[key]
        row = table[(table.width_um == key[0]) & (table.length_um == key[1])].iloc[0]
        for t2, amp, color, lw in ((c["t2_bins_ms"], c["spectrum"], "#2f6f8f", 1.2), (snr[f"{key[0]:g}_{key[1]:g}"]["t2_bins_ms"], snr[f"{key[0]:g}_{key[1]:g}"]["spectrum"], "#d9894f", 1.0)):
            t2, amp = np.asarray(t2), np.asarray(amp)
            amp = amp / amp.sum()
            keep = (t2 >= 3) & (t2 <= 5000)
            a.plot(t2[keep], np.maximum(amp[keep], 1e-5), color=color, lw=lw)  # floor = axis bottom (no path below the axes)
        a.axvline(row.tl_t2_fast_ms, color=INK, lw=0.6, ls=":")
        a.axvline(row.tl_t2_slow_ms, color=INK, lw=0.6, ls="--")
        a.axvline(cb.base.bessel_disk_t2_ms(cb.R_SMALL), color=GREY, lw=0.6, ls="-.")
        a.set(xscale="log", yscale="log", xlim=(3, 5000), ylim=(1e-5, 1), xlabel=f"T2 (ms)\nw = {key[0]:g}, L = {key[1]:g} µm ({row.regime})")
    axes[0].set_ylabel("NNLS amplitude per bin (sum = 1)")
    handles = [Line2D([], [], color="#2f6f8f", lw=1.2, label="NNLS, SNR 1000 (L-curve)"),
               Line2D([], [], color="#d9894f", lw=1.0, label="NNLS, SNR 100 (L-curve)"),
               Line2D([], [], color=INK, lw=0.6, ls=":", label="TL model fast (small-pore / exchange) mode"),
               Line2D([], [], color=INK, lw=0.6, ls="--", label="TL model slow (main) mode"),
               Line2D([], [], color=GREY, lw=0.6, ls="-.", label="Isolated small pore T2 = 430 ms")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=1.0)
    nfp.export(fig, FIG / "fig4_spectra_snr.png", multi_panel=True)


def fig_nonuniqueness(back):
    fig, ax = plt.subplots(figsize=(88 * MM, 72 * MM))
    r_s = np.linspace(0.0, 17.0, 400)

    def r_big(r_eff):
        # (r_B^2 + r_S^2) / (r_B + r_S) = R_eff  ->  r_B^2 - R_eff r_B + r_S^2 - R_eff r_S = 0
        disc = r_eff**2 - 4 * (r_s**2 - r_eff * r_s)
        return np.where(disc >= 0, 0.5 * (r_eff + np.sqrt(np.maximum(disc, 0))), np.nan)

    true_eff = back.R_eff_true_um.iloc[0]
    ax.plot(r_s, r_big(true_eff), color=INK, lw=1.1)
    for color, (_, row) in zip(("#a23b3b", "#7a4f9a"), back[back.regime == "III"].iterrows()):
        r_eff = cb.naive_radius(row.slow_t2lm_ms)
        ax.plot(r_s, r_big(r_eff), color=color, lw=1.0, ls="--")
    ax.scatter([cb.R_SMALL], [cb.R_BIG], s=36, color="#2f6f8f", edgecolor=INK, lw=0.4, zorder=4)
    ax.scatter([0.0], [true_eff], s=36, marker="s", color="#d9894f", edgecolor=INK, lw=0.4, zorder=4, clip_on=False)
    ax.set(xlim=(0, 17), ylim=(16, 22), xlabel="Small-pore radius r_S (µm)", ylabel="Big-pore radius r_B (µm)")
    rows = back[back.regime == "III"]
    handles = [Line2D([], [], color=INK, lw=1.1, label=f"All (r_S, r_B) with the true R_eff = {true_eff:.1f} µm")]
    handles += [Line2D([], [], color=c, lw=1.0, ls="--", label=f"Measured main peak, {case_label(r.width_um, r.length_um)}: R_eff = {cb.naive_radius(r.slow_t2lm_ms):.1f} µm")
                for c, (_, r) in zip(("#a23b3b", "#7a4f9a"), rows.iterrows())]
    handles += [Line2D([], [], ls="", marker="o", color="#2f6f8f", markeredgecolor=INK, ms=5.5, label="True geometry (5, 20) µm"),
                Line2D([], [], ls="", marker="s", color="#d9894f", markeredgecolor=INK, ms=5.5, label="Single-pore reading: small pore ignored")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    nfp.export(fig, FIG / "fig5_nonuniqueness_fast_exchange.png", multi_panel=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    nfp.style()
    FIG.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(OUT / "forward_models_vs_simulation.csv")
    back = pd.read_csv(OUT / "backcalculation_representative_cases.csv")
    snr = json.loads((OUT / "snr100_spectra.json").read_text(encoding="utf-8"))
    cases = cb.load_cases()
    fig_regimes(table)
    fig_validation(table, cases)
    fig_backcalc(back)
    fig_spectra(cases, snr, table)
    fig_nonuniqueness(back)


if __name__ == "__main__":
    main()
