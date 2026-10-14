"""Compute every number quoted in the coupled-pore back-calculation report and slides.

Writes, next to this file:
  generated_tables.tex   regime table (20 cases), representative-case table, back-calculation tables
  generated_worked.tex   worked substitution for the representative cases
  numbers.json           the same values for the PPT formula images
Run from the NMR package root:  python NMR-SIP/docs/coupled_pore_backcalc/make_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "advanced_tools").is_dir())  # NMR package root
sys.path.insert(0, str(ROOT))

import advanced_tools.coupled_two_pore_backcalc as cb  # noqa: E402

OUT = cb.OUT
REP = cb.REPRESENTATIVE
CASE_ID = {k: f"C{i + 1}" for i, k in enumerate(REP)}


def sci(x, digits=4):
    """LaTeX scientific notation with `digits` significant figures."""
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else "--"
    e = int(math.floor(math.log10(abs(x))))
    if -2 <= e <= 3:
        return f"{x:.{max(digits - 1 - e, 0)}f}"
    return f"{x / 10**e:.{digits - 1}f}\\times10^{{{e}}}"


def amp(x):
    """Amplitudes: 4 decimals, or 3 significant figures in scientific notation below 1e-3."""
    return f"{x:.4f}" if abs(x) >= 1e-3 else sci(x, 3)


def case_numbers(key, table):
    w, length_nom = key
    row = table[(table.width_um == w) & (table.length_um == length_nom)].iloc[0]
    h = 0.5 * w
    length = row.throat_length_mesh_um
    a_s, p_s_wall, r_s = cb.disk(cb.R_SMALL, h)
    a_b, p_b_wall, r_b = cb.disk(cb.R_BIG, h)
    rate_t = cb.throat_rate(w)
    qs = cb.quasi_static(a_s, a_b, r_s, r_b, w, length)
    p_s = a_s / (a_s + a_b)
    lf, ls, af = cb.two_site(qs["rate_s_eff"], qs["rate_b_eff"], qs["k_s"], qs["k_b"], p_s)
    mean = p_s * qs["rate_s_eff"] + (1 - p_s) * qs["rate_b_eff"]
    a_coef, d_coef = qs["rate_s_eff"] + qs["k_s"], qs["rate_b_eff"] + qs["k_b"]
    disc = math.sqrt((a_coef - d_coef) ** 2 + 4 * qs["k_s"] * qs["k_b"])
    return {
        "id": CASE_ID[key], "w": w, "L": length_nom, "Lm": length, "h": h,
        "mouth_s": 2 * cb.R_SMALL * math.asin(h / cb.R_SMALL), "mouth_b": 2 * cb.R_BIG * math.asin(h / cb.R_BIG),
        "A_S": a_s, "A_B": a_b, "P_S": p_s_wall, "P_B": p_b_wall, "R_S": r_s, "R_B": r_b, "p_S": p_s,
        "R_t": rate_t, "q": qs["q_per_um"], "qL": qs["qL"], "G": cb.D * w / length, "G_x": qs["G_x"], "G_s": qs["G_s"],
        "Rp_S": qs["rate_s_eff"], "Rp_B": qs["rate_b_eff"], "k_S": qs["k_s"], "k_B": qs["k_b"],
        "Psi": row.Psi, "Psi_eff": row.Psi_eff, "a": a_coef, "d": d_coef, "disc": disc,
        "lam_f": lf, "lam_s": ls, "T2f": 1 / lf, "T2s": 1 / ls, "Rbar": mean, "A_f": af, "eta_qs": af / p_s,
        "tl_T2f": row.tl_t2_fast_ms, "tl_T2s": row.tl_t2_slow_ms, "tl_Af": row.tl_amp_fast, "eta": row.eta_retained,
        "nn_T2f": row.nnls_t2_fast_lm_ms, "nn_Af": row.nnls_amp_fast, "nn_T2s": row.nnls_t2_slow_lm_ms,
        "regime": row.regime, "decay_err": row.tl_decay_max_abs_err,
        "asym_A": p_s * (1 - p_s) ** 3 / row.Psi_eff**2, "asym_T2f": (1 - p_s) * a_s / qs["G_x"],
    }


def write_tables(table, back, low, nums):
    lines = []
    # 1. Regime table for all 20 cases
    lines.append(r"\begin{table}[htbp]\centering\small")
    lines.append(r"\caption{20 个算例的无量纲数、分区与小孔模态（TL 模型 vs NNLS，SNR 1000）}\label{tab:regimes}")
    lines.append(r"\begin{tabular}{rrrrrrcrrrr}\toprule")
    lines.append(r"$w$ & $L$ & $\Psi$ & $qL$ & $\Psi_{\mathrm{eff}}$ & $\eta$ & 分区 & \multicolumn{2}{c}{TL：$T_{2,f}$ / $A_f$} & \multicolumn{2}{c}{NNLS：$T_{2\mathrm{lm}}$ / 分数} \\")
    lines.append(r"(\si{\micro m}) & (\si{\micro m}) & & & & & & (ms) & & (ms) & \\ \midrule")
    for _, r in table.sort_values(["width_um", "length_um"]).iterrows():
        nn_t = f"{r.nnls_t2_fast_lm_ms:.1f}" if math.isfinite(r.nnls_t2_fast_lm_ms) else "--"
        lines.append(f"{r.width_um:g} & {r.length_um:g} & {r.Psi:.3g} & {r.qL:.3g} & {r.Psi_eff:.3g} & {r.eta_retained:.3f} & {r.regime} & "
                     f"{r.tl_t2_fast_ms:.1f} & {r.tl_amp_fast:.4f} & {nn_t} & {r.nnls_amp_fast:.4f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")

    # 2. Representative cases: quasi-static (hand) vs TL exact vs NNLS
    cols = [nums[k] for k in REP]
    lines.append(r"\begin{table}[htbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\caption{代表性算例的代入求解结果：准静态两点交换（手算）、TL 精确根与 NNLS（速率单位 \si{ms^{-1}}，导纳单位 \si{\micro m^2/ms}）}\label{tab:rep}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "r" * len(cols) + r"}\toprule")
    lines.append("量 & " + " & ".join(f"{c['id']}（{c['regime']}）" for c in cols) + r" \\ \midrule")
    spec = [
        (r"$w$, $L$（\si{\micro m}）", lambda c: f"{c['w']:g}, {c['L']:g}"),
        (r"$L_m$（网格壁到壁，\si{\micro m}）", lambda c: f"{c['Lm']:.3f}"),
        (r"$P_S$, $P_B$（\si{\micro m}）", lambda c: f"{c['P_S']:.2f}, {c['P_B']:.2f}"),
        (r"$R_S$", lambda c: f"${sci(c['R_S'])}$"),
        (r"$R_B$", lambda c: f"${sci(c['R_B'])}$"),
        (r"$R_t$", lambda c: f"${sci(c['R_t'])}$"),
        (r"$q$（\si{\micro m^{-1}}）, $qL$", lambda c: f"{c['q']:.4f}, {c['qL']:.3f}"),
        (r"$G_x$", lambda c: f"${sci(c['G_x'])}$"),
        (r"$G_s$", lambda c: f"${sci(c['G_s'])}$"),
        (r"$R'_S$", lambda c: f"${sci(c['Rp_S'])}$"),
        (r"$R'_B$", lambda c: f"${sci(c['Rp_B'])}$"),
        (r"$k_S$", lambda c: f"${sci(c['k_S'])}$"),
        (r"$\Psi_{\mathrm{eff}}$", lambda c: f"{c['Psi_eff']:.4g}"),
        (r"手算 $T_{2,f}$ / $T_{2,s}$（ms）", lambda c: f"{c['T2f']:.1f} / {c['T2s']:.1f}"),
        (r"手算 $A_f$", lambda c: f"${amp(c['A_f'])}$"),
        (r"TL $T_{2,f}$ / $T_{2,s}$（ms）", lambda c: f"{c['tl_T2f']:.1f} / {c['tl_T2s']:.1f}"),
        (r"TL $A_f$（$\eta$）", lambda c: f"${amp(c['tl_Af'])}$ (${amp(c['eta'])}$)"),
        (r"NNLS $T_{2\mathrm{lm}}$：小峰 / 主峰（ms）", lambda c: f"{c['nn_T2f']:.1f} / {c['nn_T2s']:.1f}"),
        (r"NNLS 小峰分数", lambda c: f"{c['nn_Af']:.4f}"),
        (r"TL 衰减最大误差 $|\Delta M/M_0|$", lambda c: f"{c['decay_err']:.4f}"),
    ]
    for name, fn in spec:
        lines.append(name + " & " + " & ".join(fn(c) for c in cols) + r" \\")
    lines.append(r"\bottomrule\end{tabular}}\end{table}")

    # 3. Back-calculation table (SNR 1000)
    lines.append(r"\begin{table}[htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{代表性算例的反算孔径（\si{\micro m}，SNR 1000；真值 $r_S=5$、$r_B=20$）}\label{tab:back}")
    lines.append(r"\begin{tabular}{lcccccc}\toprule")
    lines.append(r"算例 & 分区 & 路线 A & 路线 B & 路线 C & 路线 D & 路线 D（精确模态） \\")
    lines.append(r" & & $r_S$ / $r_B$ & $r_S$ / $r_B$ & $r_S$ / $r_B$ & $r_S$ / $r_B$ & $r_S$ / $r_B$ \\ \midrule")
    for _, r in back.iterrows():
        key = (r.width_um, r.length_um)
        f = lambda a, b: f"{a:.2f} / {b:.2f}"
        lines.append(f"{CASE_ID[key]}（$w$={r.width_um:g}, $L$={r.length_um:g}） & {r.regime} & {f(r.A_r_small, r.A_r_big)} & {f(r.B_r_small, r.B_r_big)} & "
                     f"{f(r.C_r_small, r.C_r_big)} & {f(r.D_r_small, r.D_r_big)} & {f(r.D_exactobs_r_small, r.D_exactobs_r_big)} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")

    # 4. SNR 100
    lines.append(r"\begin{table}[htbp]\centering\small")
    lines.append(r"\caption{同一算例在 SNR 100 下的 NNLS 小峰与反算（\si{\micro m}）}\label{tab:snr}")
    lines.append(r"\begin{tabular}{lcccccc}\toprule")
    lines.append(r"算例 & 小峰 $T_{2\mathrm{lm}}$ (ms) & 小峰分数 & A：$r_S$ & B：$r_S$ & D：$r_S$ / $r_B$ & 主峰按 A：$r_B$ \\ \midrule")
    for _, r in low.iterrows():
        key = (r.w, r.L)
        if math.isfinite(r.get("fast_t2", float("nan"))):
            lines.append(f"{CASE_ID[key]} & {r.fast_t2:.1f} & {r.frac:.4f} & {r.A_rS:.2f} & {r.B_rS:.2f} & {r.D_rS:.2f} / {r.D_rB:.2f} & {r.slow_rB_naive:.2f} \\\\")
        else:
            lines.append(f"{CASE_ID[key]} & 无 & 0 & -- & -- & -- & {r.slow_rB_naive:.2f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    (HERE / "generated_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_worked(nums, back):
    c = nums[(1.0, 15.0)]
    f = nums[(4.0, 2.0)]
    s = nums[(0.1, 30.0)]
    b1 = back[(back.width_um == 0.1) & (back.length_um == 30.0)].iloc[0]
    b3 = back[(back.width_um == 1.0) & (back.length_um == 15.0)].iloc[0]
    b5 = back[(back.width_um == 4.0) & (back.length_um == 2.0)].iloc[0]
    rho, tb = cb.RHO, cb.T2B
    lines = [
        r"\subsection{算例 C3（$w=\SI{1}{\micro m}$，$L=\SI{15}{\micro m}$，区 II）的逐步代入}",
        r"\textbf{第 1 步：几何与孔体速率。}网格中喉道壁到壁长度 $L_m=" + f"{c['Lm']:.3f}" + r"$~\si{\micro m}，喉口弧长 $2R\arcsin(h/R)$ 从孔壁中扣除：",
        r"\begin{align*}",
        rf"A_S&=\pi\times5^2={c['A_S']:.3f}, & P_S&=2\pi\times5-{c['mouth_s']:.4f}={c['P_S']:.3f},\\",
        rf"A_B&=\pi\times20^2={c['A_B']:.2f}, & P_B&=2\pi\times20-{c['mouth_b']:.4f}={c['P_B']:.3f},\\",
        rf"R_S&=\frac{{{rho}\times{c['P_S']:.3f}}}{{{c['A_S']:.3f}}}+\frac{{1}}{{{tb:.0f}}}={sci(c['R_S'])}\ \mathrm{{ms^{{-1}}}}, & R_B&={sci(c['R_B'])}\ \mathrm{{ms^{{-1}}}}.",
        r"\end{align*}",
        r"\textbf{第 2 步：喉道。}$h=\SI{0.5}{\micro m}$：",
        r"\[",
        rf"R_t=\frac{{{rho}}}{{0.5}}+\frac{{1}}{{{tb:.0f}}}={sci(c['R_t'])},\qquad q=\sqrt{{R_t/D}}=\sqrt{{{sci(c['R_t'])}/2.0}}={c['q']:.5f}\ \mathrm{{\mu m^{{-1}}}},\qquad qL={c['qL']:.4f}.",
        r"\]",
        r"\textbf{第 3 步：交换导纳与汇导纳（式 \eqref{eq:gx}）。}",
        r"\[",
        rf"G_x=\frac{{Dwq}}{{\sinh qL}}=\frac{{2.0\times1\times{c['q']:.5f}}}{{\sinh {c['qL']:.4f}}}={sci(c['G_x'])},\qquad",
        rf"G_s=Dwq\tanh\frac{{qL}}{{2}}={sci(c['G_s'])}\ \mathrm{{\mu m^2/ms}}.",
        r"\]",
        r"\textbf{第 4 步：有效两点交换参数。}",
        r"\begin{align*}",
        rf"R'_S&=R_S+\frac{{G_s}}{{A_S}}={sci(c['R_S'])}+{sci(c['G_s'] / c['A_S'])}={sci(c['Rp_S'])}, & R'_B&={sci(c['Rp_B'])},\\",
        rf"k_S&=\frac{{G_x}}{{A_S}}={sci(c['k_S'])}, & k_B&=\frac{{G_x}}{{A_B}}={sci(c['k_B'])},\\",
        rf"\Psi_{{\mathrm{{eff}}}}&=\frac{{k_S}}{{R'_S-R'_B}}=\frac{{{sci(c['k_S'])}}}{{{sci(c['Rp_S'] - c['Rp_B'])}}}={c['Psi_eff']:.4f}. &&",
        r"\end{align*}",
        r"\textbf{第 5 步：本征速率（式 \eqref{eq:lam}）。}$a=R'_S+k_S=" + sci(c['a']) + r"$，$d=R'_B+k_B=" + sci(c['d']) + r"$：",
        r"\begin{align*}",
        rf"\lambda_{{f,s}}&=\frac{{a+d\pm\sqrt{{(a-d)^2+4k_Sk_B}}}}{{2}}=\frac{{{sci(c['a'] + c['d'])}\pm{sci(c['disc'])}}}{{2}}\\",
        rf"&\Rightarrow\;T_{{2,f}}=1/\lambda_f={c['T2f']:.1f}\ \mathrm{{ms}},\qquad T_{{2,s}}=1/\lambda_s={c['T2s']:.1f}\ \mathrm{{ms}}.",
        r"\end{align*}",
        r"\textbf{第 6 步：快模态振幅（式 \eqref{eq:amp}）。}",
        r"\[",
        rf"\bar R'=p_SR'_S+p_BR'_B={sci(c['Rbar'])},\qquad A_f=\frac{{\bar R'-\lambda_s}}{{\lambda_f-\lambda_s}}={c['A_f']:.4f},\qquad \eta=\frac{{A_f}}{{p_S}}={c['eta_qs']:.3f}.",
        r"\]",
        r"\textbf{第 7 步：对照。}TL 精确根给出 $T_{2,f}=" + f"{c['tl_T2f']:.1f}" + r"$~ms、$A_f=" + f"{c['tl_Af']:.4f}" + r"$；",
        r"NNLS（L-curve，SNR 1000）小峰 $T_{2\mathrm{lm}}=" + f"{c['nn_T2f']:.1f}" + r"$~ms、分数 " + f"{c['nn_Af']:.4f}" + r"；"
        r"手算与精确根的差别来自准静态近似中 $q$ 取 $\lambda=0$。",
        "",
        r"\subsection{快交换算例 C5（$w=\SI{4}{\micro m}$，$L=\SI{2}{\micro m}$，区 III）的渐近式}",
        r"\[",
        rf"\Psi_{{\mathrm{{eff}}}}={f['Psi_eff']:.2f},\qquad A_f\approx\frac{{p_Sp_B^3}}{{\Psi_{{\mathrm{{eff}}}}^2}}=\frac{{{f['p_S']:.4f}\times{1 - f['p_S']:.4f}^3}}{{{f['Psi_eff']:.2f}^2}}={sci(f['asym_A'], 3)}"
        rf"\quad(\text{{TL 精确}}\ {sci(f['tl_Af'], 3)}),",
        r"\]",
        r"\[",
        rf"T_{{2,f}}\approx\frac{{p_BA_S}}{{G_x}}=\frac{{{1 - f['p_S']:.4f}\times{f['A_S']:.2f}}}{{{f['G_x']:.4f}}}={f['asym_T2f']:.1f}\ \mathrm{{ms}}\quad(\text{{TL}}\ {f['tl_T2f']:.1f}\ \mathrm{{ms}},\ \text{{NNLS}}\ {f['nn_T2f']:.1f}\ \mathrm{{ms}}).",
        r"\]",
        r"快模态的 $T_2$ 由交换速率决定，而不是由小孔的 $S/V$ 决定；若按孤立孔公式换算，会得到一个并不存在的 "
        + f"{b5.A_r_small:.2f}" + r"~\si{\micro m} 的“孔”。",
        "",
        r"\subsection{反算的代入示例}",
        r"\textbf{路线 A（C3）：}",
        r"\[",
        rf"r_S=\frac{{2\rho_2}}{{1/T_{{2,f}}-1/T_{{2B}}}}=\frac{{0.01}}{{1/{b3.fast_t2lm_ms:.2f}-1/3000}}={b3.A_r_small:.2f}\ \mu\mathrm{{m}},\qquad "
        rf"r_B=\frac{{0.01}}{{1/{b3.slow_t2lm_ms:.2f}-1/3000}}={b3.A_r_big:.2f}\ \mu\mathrm{{m}}.",
        r"\]",
        r"\textbf{路线 B（C1）：}小峰分数 $f_S=" + f"{b1.fast_fraction:.4f}" + r"$，主峰按路线 A 得 $r_B=" + f"{b1.B_r_big:.2f}" + r"$~\si{\micro m}：",
        r"\[",
        rf"r_S=r_B\sqrt{{\frac{{f_S}}{{1-f_S}}}}={b1.B_r_big:.2f}\times\sqrt{{\frac{{{b1.fast_fraction:.4f}}}{{{1 - b1.fast_fraction:.4f}}}}}={b1.B_r_small:.2f}\ \mu\mathrm{{m}}.",
        r"\]",
        r"\textbf{区 III 的有效半径（C5）：}主峰 $T_{2\mathrm{lm}}=" + f"{b5.slow_t2lm_ms:.2f}" + r"$~ms，",
        r"\[",
        rf"R_{{\mathrm{{eff}}}}=\frac{{0.01}}{{1/{b5.slow_t2lm_ms:.2f}-1/3000}}={cb.naive_radius(b5.slow_t2lm_ms):.2f}\ \mu\mathrm{{m}}"
        rf"\quad\text{{vs.}}\quad\frac{{r_B^2+r_S^2}}{{r_B+r_S}}=\frac{{400+25}}{{25}}={b5.R_eff_true_um:.2f}\ \mu\mathrm{{m}}.",
        r"\]",
    ]
    (HERE / "generated_worked.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    table = pd.read_csv(OUT / "forward_models_vs_simulation.csv")
    back = pd.read_csv(OUT / "backcalculation_representative_cases.csv")
    low = pd.read_csv(OUT / "backcalculation_snr100.csv")
    nums = {k: case_numbers(k, table) for k in REP}
    write_tables(table, back, low, nums)
    write_worked(nums, back)
    payload = {"cases": {nums[k]["id"]: nums[k] for k in REP},
               "back": back.to_dict(orient="records"), "snr100": low.to_dict(orient="records"),
               "regime_counts": table.regime.value_counts().to_dict(),
               "max_tl_decay_err": float(table.tl_decay_max_abs_err.max())}
    (HERE / "numbers.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print("ok", payload["regime_counts"], payload["max_tl_decay_err"])


if __name__ == "__main__":
    main()
