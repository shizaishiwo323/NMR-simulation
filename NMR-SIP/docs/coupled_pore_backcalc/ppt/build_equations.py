"""Typeset the slide formulas with XeLaTeX (one page each) and rasterize them to transparent 600-dpi PNGs.

Substituted values come from ../numbers.json (written by ../make_numbers.py), so slides and PDF agree.
Writes eq/<key>.png and eq/sizes.json (natural size in pt).
"""

import json
import math
import subprocess
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
XELATEX = r"C:\Users\imgw\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
NUM = json.loads((HERE.parent / "numbers.json").read_text(encoding="utf-8"))
C3, C5 = NUM["cases"]["C3"], NUM["cases"]["C5"]
BACK = {(round(r["width_um"], 3), round(r["length_um"], 3)): r for r in NUM["back"]}
B1, B3, B5 = BACK[(0.1, 30.0)], BACK[(1.0, 15.0)], BACK[(4.0, 2.0)]
MS, IMS, UM = r"\ \mathrm{ms}", r"\ \mathrm{ms^{-1}}", r"\ \mu\mathrm{m}"


def sci(x, digits=4):
    e = int(math.floor(math.log10(abs(x))))
    if -2 <= e <= 3:
        return f"{x:.{max(digits - 1 - e, 0)}f}"
    return f"{x / 10**e:.{digits - 1}f}\\times10^{{{e}}}"


def naive(t2):
    return 0.01 / (1 / t2 - 1 / 3000)


EQ = {
    # model
    "node": r"A_i\frac{\mathrm dc_i}{\mathrm dt}=-R_iA_ic_i-J_i,\qquad R_i=\frac{\rho_2P_i}{A_i}+\frac{1}{T_{2B}},\quad i\in\{S,B\}",
    "throat": r"\frac{\partial u}{\partial t}=D\frac{\partial^2u}{\partial x^2}-R_tu,\qquad R_t=\frac{\rho_2}{h}+\frac{1}{T_{2B}},\qquad u(0)=c_S,\ u(L)=c_B",
    "mode": r"U(x)=\frac{c_S\sinh q(L-x)+c_B\sinh qx}{\sinh qL},\qquad J_S=Dwq\left(c_S\coth qL-c_B\,\mathrm{csch}\,qL\right),\qquad q^2=\frac{R_t-\lambda}{D}",
    "char": r"\boxed{\;\big[(R_S-\lambda)A_SS+DwC\big]\big[(R_B-\lambda)A_BS+DwC\big]-(Dw)^2=0,\quad S=\frac{\sinh qL}{q},\ C=\cosh qL\;}",
    "amp_n": r"A_n=\frac{\left(\int_\Omega\phi_n\,\mathrm d\Omega\right)^2}{|\Omega|\int_\Omega\phi_n^2\,\mathrm d\Omega},\qquad \int_\Omega\phi_n=A_Sc_S+A_Bc_B+w\int_0^LU\,\mathrm dx",
    "gx": r"G_x=\frac{Dwq}{\sinh qL},\qquad G_s=Dwq\tanh\frac{qL}{2},\qquad q\simeq\sqrt{R_t/D}\quad(\lambda\ll R_t)",
    "twosite": r"\frac{\mathrm d}{\mathrm dt}\begin{pmatrix}m_S\\ m_B\end{pmatrix}=-\begin{pmatrix}R'_S+k_S&-k_B\\-k_S&R'_B+k_B\end{pmatrix}\begin{pmatrix}m_S\\ m_B\end{pmatrix},\qquad R'_i=R_i+\frac{G_s}{A_i},\quad k_i=\frac{G_x}{A_i}",
    "lim_small": r"qL\ll1:\quad G_x\to\frac{Dw}{L},\qquad G_s\to\tfrac12\,wLR_t",
    "lim_large": r"qL\gg1:\quad G_x\to2Dwq\,e^{-qL}\to0,\qquad G_s\to w\sqrt{DR_t}",
    "lam": r"\lambda_{f,s}=\frac{a+d\pm\sqrt{(a-d)^2+4k_Sk_B}}{2},\qquad a=R'_S+k_S,\quad d=R'_B+k_B",
    "ampf": r"A_f=\frac{\bar R'-\lambda_s}{\lambda_f-\lambda_s},\qquad \bar R'=p_SR'_S+p_BR'_B,\qquad A_s=1-A_f",
    "dimless": r"\Psi=\frac{Dw/L}{A_S(R_S-R_B)},\qquad \Psi_{\mathrm{eff}}=\frac{k_S}{R'_S-R'_B},\qquad \eta=\frac{A_f}{p_S}",
    "slow": r"\Psi_{\mathrm{eff}}\to0:\quad \lambda_f\to R'_S,\quad \lambda_s\to R'_B,\quad A_f\to p_S",
    "fast": r"\Psi_{\mathrm{eff}}\gg1:\quad \lambda_s\to\bar R',\quad \lambda_f\to\frac{k_S}{p_B},\quad A_f\approx\frac{p_Sp_B^3}{\Psi_{\mathrm{eff}}^2}",
    "reff": r"R_{\mathrm{eff}}=\frac{2\rho_2}{\bar R-1/T_{2B}}=\frac{2(A_S+A_B)}{P_S+P_B}=\frac{r_B^2+r_S^2}{r_B+r_S}",
    "routeA": r"r=\frac{2\rho_2}{1/T_{2\mathrm{lm}}-1/T_{2B}}",
    "routeB": r"p_S=\frac{r_S^2}{r_S^2+r_B^2}\approx f_S\;\Longrightarrow\;r_S=r_B\sqrt{f_S/f_B}",
    "routeC": r"(r_S,r_B,G)\;\leftarrow\;(\lambda_f,\lambda_s,A_f)\quad\text{两点交换（无汇）}",
    "routeD": r"\min_{r_S,r_B}\ \big\|(\ln\lambda_f,\ln\lambda_s,\ln A_f)_{\mathrm{TL}}-(\cdot)_{\mathrm{NNLS}}\big\|,\quad w,L\ \text{已知}",
    # worked C3
    "c3_1": rf"A_S={C3['A_S']:.3f},\ P_S={C3['P_S']:.3f},\ A_B={C3['A_B']:.2f},\ P_B={C3['P_B']:.3f}\;\Rightarrow\;R_S={sci(C3['R_S'])},\ R_B={sci(C3['R_B'])}" + IMS,
    "c3_2": rf"R_t=\frac{{0.005}}{{0.5}}+\frac{{1}}{{3000}}={sci(C3['R_t'])},\qquad q=\sqrt{{R_t/D}}={C3['q']:.5f}\ \mu\mathrm{{m}}^{{-1}},\qquad qL={C3['qL']:.4f}",
    "c3_3": rf"G_x=\frac{{2.0\times1\times{C3['q']:.5f}}}{{\sinh {C3['qL']:.4f}}}={sci(C3['G_x'])},\qquad G_s=Dwq\tanh\frac{{qL}}{{2}}={sci(C3['G_s'])}\ \mu\mathrm{{m^2/ms}}",
    "c3_4": rf"R'_S={sci(C3['R_S'])}+\frac{{{sci(C3['G_s'])}}}{{{C3['A_S']:.2f}}}={sci(C3['Rp_S'])},\quad R'_B={sci(C3['Rp_B'])},\quad k_S={sci(C3['k_S'])},\quad k_B={sci(C3['k_B'])}",
    "c3_5": rf"\Psi_{{\mathrm{{eff}}}}=\frac{{k_S}}{{R'_S-R'_B}}=\frac{{{sci(C3['k_S'])}}}{{{sci(C3['Rp_S'] - C3['Rp_B'])}}}={C3['Psi_eff']:.4f}\quad(\text{{区 II}})",
    "c3_6": rf"\lambda_{{f,s}}=\frac{{{sci(C3['a'] + C3['d'])}\pm{sci(C3['disc'])}}}{{2}}\;\Rightarrow\;T_{{2,f}}={C3['T2f']:.1f}" + MS + rf",\quad T_{{2,s}}={C3['T2s']:.1f}" + MS,
    "c3_7": rf"\bar R'={sci(C3['Rbar'])},\qquad A_f=\frac{{\bar R'-\lambda_s}}{{\lambda_f-\lambda_s}}={C3['A_f']:.4f},\qquad \eta={C3['eta_qs']:.3f}",
    "c3_8": rf"\text{{TL：}}T_{{2,f}}={C3['tl_T2f']:.1f}" + MS + rf",\ A_f={C3['tl_Af']:.4f};\qquad \text{{NNLS：}}T_{{2\mathrm{{lm}}}}={C3['nn_T2f']:.1f}" + MS + rf",\ {C3['nn_Af']:.4f}",
    # fast-exchange C5
    "c5_1": rf"\Psi_{{\mathrm{{eff}}}}={C5['Psi_eff']:.2f},\qquad A_f\approx\frac{{p_Sp_B^3}}{{\Psi_{{\mathrm{{eff}}}}^2}}={sci(C5['asym_A'], 3)}\quad(\text{{TL}}\ {sci(C5['tl_Af'], 3)})",
    "c5_2": rf"T_{{2,f}}\approx\frac{{p_BA_S}}{{G_x}}=\frac{{{1 - C5['p_S']:.4f}\times{C5['A_S']:.2f}}}{{{C5['G_x']:.4f}}}={C5['asym_T2f']:.1f}" + MS + rf"\quad(\text{{TL}}\ {C5['tl_T2f']:.1f},\ \text{{NNLS}}\ {C5['nn_T2f']:.1f}" + MS + ")",
    "c5_3": rf"r_{{\text{{伪}}}}=\frac{{0.01}}{{1/{B5['fast_t2lm_ms']:.2f}-1/3000}}={B5['A_r_small']:.2f}" + UM + r"\quad(\text{真值}\ 5\ \mu\mathrm{m})",
    # worked back-calculation
    "bA": rf"r_S=\frac{{0.01}}{{1/{B3['fast_t2lm_ms']:.2f}-1/3000}}={B3['A_r_small']:.2f}" + UM + rf",\qquad r_B=\frac{{0.01}}{{1/{B3['slow_t2lm_ms']:.2f}-1/3000}}={B3['A_r_big']:.2f}" + UM,
    "bB": rf"r_S={B1['B_r_big']:.2f}\times\sqrt{{\frac{{{B1['fast_fraction']:.4f}}}{{{1 - B1['fast_fraction']:.4f}}}}}={B1['B_r_small']:.2f}" + UM,
    "bR": rf"R_{{\mathrm{{eff}}}}=\frac{{0.01}}{{1/{B5['slow_t2lm_ms']:.2f}-1/3000}}={naive(B5['slow_t2lm_ms']):.2f}" + UM + rf"\quad\text{{vs.}}\quad\frac{{400+25}}{{25}}={B5['R_eff_true_um']:.2f}" + UM,
}

PREAMBLE = r"""\documentclass[border=3pt,multi=eqpage]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{xeCJK}
\setCJKmainfont{Microsoft YaHei}
\newenvironment{eqpage}{$\displaystyle}{$}
\begin{document}
"""


def main() -> None:
    out = HERE / "eq"
    out.mkdir(exist_ok=True)
    keys = list(EQ)
    tex = PREAMBLE + "\n".join(r"\begin{eqpage}" + EQ[k] + r"\end{eqpage}" for k in keys) + "\n\\end{document}\n"
    (out / "equations.tex").write_text(tex, encoding="utf-8")
    subprocess.run([XELATEX, "-interaction=nonstopmode", "-halt-on-error", "equations.tex"], cwd=out, check=True, stdout=subprocess.DEVNULL)
    doc = pymupdf.open(out / "equations.pdf")
    assert len(doc) == len(keys), (len(doc), len(keys))
    sizes = {}
    for key, page in zip(keys, doc):
        page.get_pixmap(dpi=600, alpha=True).save(out / f"{key}.png")
        sizes[key] = [round(page.rect.width, 3), round(page.rect.height, 3)]
    (out / "sizes.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    print(f"{len(keys)} equations")


if __name__ == "__main__":
    main()
