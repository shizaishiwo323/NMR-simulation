"""Intuition figures for the C3 walk-through (w = 1 um, L = 15 um), companion to coupled_pore_backcalc.pdf.

All numbers come from ../numbers.json (written by ../make_numbers.py from the 20-case simulation outputs);
the time curves use the quasi-static two-site model with those C3 parameters. Writes figures/*.png and
prints the derived values quoted in c3_intuitive.tex.

    python NMR-SIP/docs/coupled_pore_backcalc/c3_intuitive/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "advanced_tools").is_dir())  # NMR package root
sys.path.insert(0, str(ROOT))
import advanced_tools.ideal_two_pore_nature_plots as nfp  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle  # noqa: E402

OUT = HERE / "figures"
NUM = json.loads((HERE.parent / "numbers.json").read_text(encoding="utf-8").replace("NaN", "null"))
C3 = NUM["cases"]["C3"]
BACK3 = next(b for b in NUM["back"] if b["width_um"] == 1.0 and b["length_um"] == 15.0)
RHO2, D, T2B = 0.005, 2.0, 3000.0  # um/ms, um^2/ms, ms (same as the sweep)
MM = nfp.MM
C_SMALL, C_BIG, C_GREY = "#c0643b", "#2f6f8f", "#8a8a8a"


def ruler(rate: float) -> float:
    """Isolated-pore 'ruler' r = 2 rho2 / (1/T2 - 1/T2B), with the rate 1/T2 in 1/ms."""
    return 2 * RHO2 / (rate - 1 / T2B)


def style():
    nfp.style()
    matplotlib.rcParams.update({
        "font.family": ["Arial", "Microsoft YaHei"],  # per-glyph fallback: Latin in Arial, CJK in YaHei
        "axes.unicode_minus": False,
        "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        # mathtext has no CJK fallback: labels are either pure math or plain text, never mixed
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    })


FLOOR = 1e-3


def above(y):
    """Hide values below the log-axis floor (paths must stay inside the axes)."""
    return np.where(y >= FLOOR, y, np.nan)


def coupled(t):
    """Pore-averaged magnetisation c_S(t), c_B(t) with sink + exchange, both starting at 1."""
    k_s, k_b = C3["k_S"], C3["k_B"]
    K = np.array([[C3["Rp_S"] + k_s, -k_s], [-k_b, C3["Rp_B"] + k_b]])
    lam, vec = np.linalg.eig(K)
    coef = np.linalg.solve(vec, np.ones(2))
    c = (vec * coef) @ np.exp(-np.outer(lam, t))
    return c[0], c[1], lam, vec


# ---------------------------------------------------------------- figure 1: geometry + processes
def fig_schematic():
    rng = np.random.default_rng(7)
    R_B, R_S, L, w = 20.0, 5.0, 15.0, 1.0
    xs = R_B + L + R_S  # small-pore centre (to scale)
    fig, ax = plt.subplots(figsize=(170 * MM, 62 * MM))
    ax.add_patch(Rectangle((R_B - 1, -w / 2), L + 2, w, facecolor=nfp.C_THROAT, edgecolor="#a23b3b", lw=0.6, zorder=1))
    for x0, r in ((0, R_B), (xs, R_S)):
        ax.add_patch(Circle((x0, 0), r, facecolor="#dbe7f1", edgecolor=INK, lw=0.9, zorder=2))
    # illustrative random walks, reflected at the pore wall
    for x0, r, n, steps, sig in ((0, R_B, 8, 110, 1.5), (xs, R_S, 3, 70, 0.5)):
        for _ in range(n):
            ang, rad = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0, 0.8))
            p = np.array([x0 + rad * np.cos(ang), rad * np.sin(ang)])
            path = [p.copy()]
            for _ in range(steps):
                p = p + rng.normal(0, sig, 2)
                d = p - [x0, 0]
                if np.hypot(*d) > r:
                    p = np.array([x0, 0]) + d / np.hypot(*d) * r * 0.985
                path.append(p.copy())
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], color=C_GREY, lw=0.35, alpha=0.8, zorder=3)
    # wall encounters spread round each pore, away from the throat mouth (illustrative)
    for x0, r, n, lo, hi in ((0, R_B, 22, 0.3, 2 * np.pi - 0.3), (xs, R_S, 6, 0.6, 2 * np.pi - 0.6)):
        ang = rng.uniform(lo, hi, n) + (np.pi if x0 else 0)
        ax.plot(x0 + 0.985 * r * np.cos(ang), 0.985 * r * np.sin(ang), "x", color="#a23b3b", ms=3.2, mew=0.8, zorder=4)
    ax.add_patch(FancyArrowPatch((R_B + 1.2, 2.2), (R_B + L - 1.2, 2.2), arrowstyle="<|-|>", mutation_scale=7,
                                 color=INK, lw=0.8, zorder=5))
    ax.set_xlim(-R_B - 1, xs + R_S + 1)
    ax.set_ylim(-R_B - 1, R_B + 1)
    ax.set_aspect("equal")
    ax.set_xticks([-20, 0, 20, 35, 40])
    ax.set_yticks([-20, 0, 20])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    handles = [
        Patch(facecolor="#dbe7f1", edgecolor=INK, lw=0.9, label="孔隙水：大孔 B（R = 20 µm），小孔 S（R = 5 µm）"),
        Line2D([], [], color=C_GREY, lw=0.6, label="水分子的随机游走（示意）"),
        Line2D([], [], ls="none", marker="x", color="#a23b3b", ms=4, mew=0.9, label="碰到孔壁：以很小的概率被消磁（ρ2）"),
        Line2D([], [], color=INK, lw=0.8, marker=">", ms=3, label="喉道角色①走廊：两孔的水经喉道互相交换"),
        Patch(facecolor=nfp.C_THROAT, edgecolor="#a23b3b", lw=0.6, label="喉道角色②下水道：两壁很近，进来的磁化很快被消掉"),
        Patch(facecolor="none", edgecolor="none", label="（喉道 w = 1 µm、L = 15 µm，按真实比例绘制）"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5), handlelength=1.6)
    nfp.export(fig, OUT / "fig_c3_schematic.png", multi_panel=False)


INK = nfp.INK


# ---------------------------------------------------------------- figure 2: what qL means
def fig_throat():
    x = np.linspace(0, 1, 301)
    cases = [(0.3, "qL = 0.3：短喉道，近似无损通道", "#7fb3c9", "-"),
             (C3["qL"], f"qL = {C3['qL']:.2f}：C3", C_SMALL, "-"),
             (3.0, "qL = 3：长喉道，磁化走不到对面", "#2f4f6f", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 58 * MM), gridspec_kw={"wspace": 0.32})
    for qL, lab, col, ls in cases:
        axes[0].plot(x, np.cosh(qL * (x - 0.5)) / np.cosh(qL / 2), color=col, ls=ls, label=lab)
        axes[1].plot(x, np.sinh(qL * x) / np.sinh(qL), color=col, ls=ls, label=lab)
    axes[1].plot(x, x, color=C_GREY, ls="--", lw=0.9, label="喉道壁不消磁时（线性）")
    axes[0].set_ylabel("喉道内磁化（两端 = 1）")
    axes[1].set_ylabel("喉道内磁化（大孔端 = 1，小孔端 = 0）")
    for ax, tag in zip(axes, "ab"):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("沿喉道位置 x / L（0 = 小孔端，1 = 大孔端）")
        ax.text(-0.2, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    nfp.export(fig, OUT / "fig_c3_throat_profile.png", multi_panel=True)


# ---------------------------------------------------------------- figure 3: time domain
def fig_time():
    t = np.linspace(0, 3000, 601)
    c_s, c_b, lam, vec = coupled(t)
    fig, axes = plt.subplots(1, 3, figsize=(170 * MM, 58 * MM), gridspec_kw={"wspace": 0.5})
    for ax, r0, r1, cc, col in ((axes[0], C3["R_S"], C3["Rp_S"], c_s, C_SMALL), (axes[1], C3["R_B"], C3["Rp_B"], c_b, C_BIG)):
        ax.plot(t, above(np.exp(-r0 * t)), color=col, ls=":", lw=1.2, label="孤立孔")
        ax.plot(t, above(np.exp(-r1 * t)), color=col, ls="--", lw=1.0, label="+ 喉道汇")
        ax.plot(t, above(cc), color=col, lw=1.4, label="+ 汇 + 交换")
    total = C3["p_S"] * c_s + (1 - C3["p_S"]) * c_b
    order = np.argsort(lam)[::-1]  # fast first
    axes[2].plot(t, total, color=INK, lw=1.4, label="总信号 M(t)/M(0)")
    axes[2].plot(t, above(C3["A_f"] * np.exp(-lam[order[0]] * t)), color=C_SMALL, lw=1.1, ls="--", label=f"快模态（小峰）{C3['A_f']:.3f}·e^(−t/{1 / lam[order[0]]:.0f} ms)")
    axes[2].plot(t, (1 - C3["A_f"]) * np.exp(-lam[order[1]] * t), color=C_BIG, lw=1.1, ls="--", label=f"慢模态（主峰）{1 - C3['A_f']:.3f}·e^(−t/{1 / lam[order[1]]:.0f} ms)")
    labels = ("小孔 S 内平均磁化", "大孔 B 内平均磁化", "总信号（归一化）")
    for ax, tag, lab in zip(axes, "abc", labels):
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 1.2)
        ax.set_xlim(0, 3000)
        ax.set_xticks([0, 1000, 2000, 3000])
        ax.set_xlabel("时间 t (ms)")
        ax.set_ylabel(lab)
        ax.text(-0.3, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, _ = axes[1].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    pair = [(Line2D([], [], color=INK, ls=a.get_ls(), lw=a.get_lw()), l) for a, l in zip(h0, l0)]
    fig.legend([p[0] for p in pair] + h2, [f"{l}（a 小孔 / b 大孔）" for _, l in pair] + l2,
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    nfp.export(fig, OUT / "fig_c3_time_domain.png", multi_panel=True)
    return lam[order], vec[:, order]


# ---------------------------------------------------------------- figure 4: where the peaks move
def fig_shift():
    steps = ["孤立孔", "+ 喉道汇", "+ 汇\n+ 交换\n（手算）", "精确模型\n（TL）", "全二维\n模拟\n+ NNLS"]
    small = [ruler(C3["R_S"]), ruler(C3["Rp_S"]), ruler(C3["lam_f"]), ruler(1 / C3["tl_T2f"]), ruler(1 / C3["nn_T2f"])]
    big = [ruler(C3["R_B"]), ruler(C3["Rp_B"]), ruler(C3["lam_s"]), ruler(1 / C3["tl_T2s"]), ruler(1 / C3["nn_T2s"])]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), gridspec_kw={"wspace": 0.28})
    shades = ["#d9d9d9", "#b9c9d6", "#7fa6bf", "#4f86a6", "#2f6f8f"]
    for ax, vals, true, lab, top in ((axes[0], small, 5.0, "小孔：由峰位换算的半径 (µm)", 6), (axes[1], big, 20.0, "大孔：由峰位换算的半径 (µm)", 24)):
        ax.bar(range(5), vals, color=shades, width=0.68, edgecolor=INK, lw=0.5)
        ax.axhline(true, color="#a23b3b", ls="--", lw=1.0)
        ax.set_xticks(range(5), steps)
        ax.set_ylim(0, top)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelsize=6.6)
    for ax, tag in zip(axes, "ab"):
        ax.text(-0.15, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.legend([Line2D([], [], color="#a23b3b", ls="--", lw=1.0)], ["真实半径（小孔 5 µm，大孔 20 µm）"],
               loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    nfp.export(fig, OUT / "fig_c3_peak_shift.png", multi_panel=True)
    return small, big


def eta_two_site(psi, p_s, r_s=2.0, r_b=1.0):
    """Small-pore retention eta = A_f / p_S of the two-site model for given Psi_eff.

    The absolute rates r_s > r_b (= R'_S, R'_B) are free: only Psi_eff and p_S enter the result.
    """
    k_s = np.asarray(psi) * (r_s - r_b)
    k_b = k_s * p_s / (1 - p_s)
    a, d = r_s + k_s, r_b + k_b
    disc = np.sqrt((a - d) ** 2 + 4 * k_s * k_b)
    lam_f, lam_s = (a + d + disc) / 2, (a + d - disc) / 2
    rbar = p_s * r_s + (1 - p_s) * r_b
    return (rbar - lam_s) / (lam_f - lam_s) / p_s


# ---------------------------------------------------------------- figure 5: where C3 sits
def fig_regime():
    p_s = C3["p_S"]
    psi = np.logspace(-3.3, 2.3, 400)
    eta = eta_two_site(psi, p_s)
    b1, b2 = np.interp(0.8, eta[::-1], psi[::-1]), np.interp(0.1, eta[::-1], psi[::-1])
    fig, ax = plt.subplots(figsize=(110 * MM, 66 * MM))
    ax.axvspan(psi[0], b1, color="#2c6e8f", alpha=0.12, lw=0)
    ax.axvspan(b1, b2, color="#d9895b", alpha=0.15, lw=0)
    ax.axvspan(b2, psi[-1], color="#7a4f9a", alpha=0.12, lw=0)
    ax.plot(psi, np.where(eta >= 5e-4, eta, np.nan), color=INK, lw=1.2)
    for k, c in NUM["cases"].items():
        big = k == "C3"
        ax.plot(c["Psi_eff"], c["eta"], "o", ms=7 if big else 4.5, mfc=C_SMALL if big else "white", mec=C_SMALL if big else INK, mew=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(psi[0], psi[-1])
    ax.set_ylim(5e-4, 1.6)
    ax.set_xlabel(r"$\Psi_{\rm eff} = k_S\,/\,(R^{\prime}_S - R^{\prime}_B)$")
    ax.set_ylabel(r"$\eta = A_f\,/\,p_S$")
    handles = [Line2D([], [], color=INK, lw=1.2, label="两点交换模型的普适曲线"),
               Line2D([], [], ls="none", marker="o", mfc="white", mec=INK, ms=4.5, label="代表算例 C1、C2、C4、C5（自左向右）"),
               Line2D([], [], ls="none", marker="o", mfc=C_SMALL, mec=C_SMALL, ms=7, label=f"C3（本文算例）：η = {C3['eta']:.2f}"),
               Patch(facecolor="#2c6e8f", alpha=0.25, label="区 I 慢交换：两峰各自代表两孔"),
               Patch(facecolor="#d9895b", alpha=0.3, label="区 II 过渡：两峰都被拉偏"),
               Patch(facecolor="#7a4f9a", alpha=0.25, label="区 III 快交换：只剩一个主峰")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=7.4)  # subscripts stay >= 5 pt
    nfp.export(fig, OUT / "fig_c3_regime.png", multi_panel=False)
    return b1, b2


# ---------------------------------------------------------------- figure 6: why the curve is universal
def floor_mask(y, lo=1e-4):
    return np.where(y >= lo, y, np.nan)


def fig_universal():
    p_s = C3["p_S"]
    psi = np.logspace(-3, 2, 400)
    psi_pts = np.array([0.003, 0.03, 0.3, 1.0, 3.0, 30.0])
    rate_pairs = [((C3["Rp_S"], C3["Rp_B"]), "s", "C3 的实际速率 R′S = 0.00317、R′B = 0.000886（1/ms）"),
                  ((2.0, 1.0), "^", "任取 R′S = 2、R′B = 1"),
                  ((0.05, 1e-4), "D", "任取 R′S = 0.05、R′B = 0.0001")]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), gridspec_kw={"wspace": 0.3})
    axes[0].plot(psi, floor_mask(eta_two_site(psi, p_s)), color=INK, lw=1.2, label=f"普适曲线（pS = {p_s:.4f}）")
    for (r_s, r_b), mk, lab in rate_pairs:
        axes[0].plot(psi_pts, eta_two_site(psi_pts, p_s, r_s, r_b), mk, ms=4.5, mfc="none", mec=C_SMALL, mew=0.9, label=lab)
    shades = ["#9fb7cf", INK, "#c0643b", "#7a4f9a"]
    for ps_i, col in zip((0.01, p_s, 0.2, 0.5), shades):
        lab = f"pS = {ps_i:.4f}（C3）" if ps_i == p_s else f"pS = {ps_i:g}"
        axes[1].plot(psi, floor_mask(eta_two_site(psi, ps_i)), color=col, lw=1.6 if ps_i == p_s else 1.1, label=lab)
    for ax, tag in zip(axes, "ab"):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(psi[0], psi[-1])
        ax.set_ylim(1e-4, 1.6)
        ax.set_xlabel(r"$\Psi_{\rm eff}$")
        ax.set_ylabel(r"$\eta = A_f\,/\,p_S$")
        ax.text(-0.17, 1.04, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    axes[0].legend(h0, l0, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axes[1].legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    nfp.export(fig, OUT / "fig_c3_universal.png", multi_panel=True)
    return {x: float(eta_two_site(x, p_s)) for x in (0.01, 0.1, C3["Psi_eff"], 2.0, 10.0)}


def main():
    OUT.mkdir(exist_ok=True)
    style()
    fig_schematic()
    fig_throat()
    lam, vec = fig_time()
    small, big = fig_shift()
    b1, b2 = fig_regime()
    curve = fig_universal()
    q = C3["q"]
    report = {
        "P/A small, big, throat (1/um)": (C3["P_S"] / C3["A_S"], C3["P_B"] / C3["A_B"], 2 / C3["w"]),
        "T2 isolated small, big (ms)": (1 / C3["R_S"], 1 / C3["R_B"]),
        "R^2/D small, big (ms)": (25 / D, 400 / D),
        "kappa small, big": (RHO2 * 5 / D, RHO2 * 20 / D),
        "T2 throat (ms)": 1 / C3["R_t"],
        "ell = 1/q (um)": 1 / q,
        "qL, sinh, tanh(qL/2), cosh(qL/2)": (C3["qL"], np.sinh(C3["qL"]), np.tanh(C3["qL"] / 2), np.cosh(C3["qL"] / 2)),
        "Dw/L, G_x, G_x/(Dw/L)": (C3["G"], C3["G_x"], C3["G_x"] / C3["G"]),
        "G_s, G_s/A_S, G_s/A_B": (C3["G_s"], C3["G_s"] / C3["A_S"], C3["G_s"] / C3["A_B"]),
        "rate increase by sink small, big (%)": (100 * (C3["Rp_S"] / C3["R_S"] - 1), 100 * (C3["Rp_B"] / C3["R_B"] - 1)),
        "T2 with sink small, big (ms)": (1 / C3["Rp_S"], 1 / C3["Rp_B"]),
        "1/k_S, 1/k_B (ms)": (1 / C3["k_S"], 1 / C3["k_B"]),
        "Rp_S - Rp_B, Psi_eff": (C3["Rp_S"] - C3["Rp_B"], C3["Psi_eff"]),
        "a, d, a+d, disc": (C3["a"], C3["d"], C3["a"] + C3["d"], C3["disc"]),
        "lam_f, lam_s, T2f, T2s": (*lam, *(1 / lam)),
        "fast-mode c_B/c_S, slow-mode c_S/c_B": (vec[1, 0] / vec[0, 0], vec[0, 1] / vec[1, 1]),
        "Rbar, 1/Rbar (ms)": (C3["Rbar"], 1 / C3["Rbar"]),
        "A_f, eta (hand), TL A_f, TL eta, NNLS frac, NNLS eta": (C3["A_f"], C3["eta_qs"], C3["tl_Af"], C3["eta"], C3["nn_Af"], C3["nn_Af"] / C3["p_S"]),
        "ruler small steps (um)": small,
        "ruler big steps (um)": big,
        "route A, B, D (r_S, r_B)": ((BACK3["A_r_small"], BACK3["A_r_big"]), (BACK3["B_r_small"], BACK3["B_r_big"]), (BACK3["D_r_small"], BACK3["D_r_big"])),
        "regime bounds Psi_eff (eta 0.8, 0.1)": (b1, b2),
        "universal curve eta at Psi = 0.01, 0.1, C3, 2, 10": tuple(curve.values()),
        "fast-exchange asymptote p_B^3/Psi^2 at Psi = 10": (1 - C3["p_S"]) ** 3 / 100,
        "throat profile: midpoint 1/cosh(qL/2) for C3, qL=3; end slopes q*tanh(qL/2), q/sinh(qL)": (
            1 / np.cosh(C3["qL"] / 2), 1 / np.cosh(1.5), q * np.tanh(C3["qL"] / 2), q / np.sinh(C3["qL"])),
    }
    for k, v in report.items():
        flat = np.ravel(np.array(v, dtype=float))
        print(f"{k}: " + ", ".join(f"{x:.6g}" for x in flat))


if __name__ == "__main__":
    main()
