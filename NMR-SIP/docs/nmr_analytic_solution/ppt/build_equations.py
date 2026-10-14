"""Typeset every slide formula with XeLaTeX (one page per formula) and rasterize to transparent PNGs.

Writes eq/<key>.png and eq/sizes.json (natural size in pt, used to place images at a uniform scale).
"""

import json
import subprocess
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
XELATEX = r"C:\Users\imgw\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
U = r"\ \mathrm{ms}"
UI = r"\ \mathrm{ms^{-1}}"
UM = r"\ \mu\mathrm{m}"

EQ = {
    "bt": r"\frac{\partial M}{\partial t}=D\nabla^{2}M-\frac{M}{T_{2B}},\qquad \mathbf r\in\Omega",
    "robin": r"D\,\mathbf n\cdot\nabla M+\rho_2 M=0,\qquad \mathbf r\in\partial\Omega",
    "signal": r"S(t)=\int_\Omega M(\mathbf r,t)\,\mathrm d\Omega,\qquad M(\mathbf r,0)=M_0",
    "sep": r"M(\mathbf r,t)=e^{-t/T_{2B}}\,\phi_n(\mathbf r)\,e^{-t/T_{2S}}",
    "eig": r"\nabla^2\phi_n+\frac{\xi_n^2}{a^2}\phi_n=0,\qquad D\,\mathbf n\cdot\nabla\phi_n+\rho_2\phi_n=0",
    "kappa": r"\kappa=\frac{\rho_2\,a}{D}",
    "modes": r"\frac{1}{T_{2,n}}=\frac{1}{T_{2B}}+\frac{D\,\xi_n^2}{a^2},\qquad \frac{S(t)}{S(0)}=\sum_{n}A_n\,e^{-t/T_{2,n}}",
    "disk_phi": r"\phi_n(r)=J_0\!\left(\xi_n\frac{r}{R}\right)",
    "disk_char": r"\boxed{\;\xi_n\,\frac{J_1(\xi_n)}{J_0(\xi_n)}=\kappa=\frac{\rho_2R}{D}\;}",
    "disk_A": r"A_n=\frac{4\kappa^2}{\xi_n^2\left(\xi_n^2+\kappa^2\right)}",
    "slab_phi": r"\phi_n(y)=\cos\!\left(\xi_n\frac{y}{h}\right)",
    "slab_char": r"\boxed{\;\xi_n\tan\xi_n=\kappa=\frac{\rho_2h}{D}\;}",
    "slab_A": r"A_n=\frac{2\sin^2\xi_n}{\xi_n\left(\xi_n+\sin\xi_n\cos\xi_n\right)}",
    "fd_disk": r"\frac{J_1(x)}{J_0(x)}\approx\frac{x}{2}+\frac{x^3}{16}\;\Longrightarrow\;\xi_0^2\approx2\kappa\left(1-\frac{\kappa}{4}\right)",
    "fd_slab": r"x\tan x\approx x^2+\frac{x^4}{3}\;\Longrightarrow\;\xi_0^2\approx\kappa\left(1-\frac{\kappa}{3}\right)",
    "fd_main": r"\boxed{\;\frac{1}{T_2}\approx\frac{1}{T_{2B}}+\rho_2\frac{S}{V}\;},\qquad \frac{S}{V}=\begin{cases}2/R, & \text{二维圆盘}\\ 1/h, & \text{二维狭缝}\end{cases}",
    "inverse": r"R=\frac{2\rho_2}{1/T_2-1/T_{2B}},\qquad h=\frac{\rho_2}{1/T_2-1/T_{2B}}",
    "b1": r"\kappa=\frac{\rho_2R}{D}=\frac{0.005\times20}{2.0}=0.05\ll1",
    "b2": r"\xi_0\frac{J_1(\xi_0)}{J_0(\xi_0)}=0.05\;\Longrightarrow\;\xi_0=0.314262,\quad \xi_0^2=0.0987604",
    "b3": r"\frac{1}{T_{2S}}=\frac{D\xi_0^2}{R^2}=\frac{2.0\times0.0987604}{20^2}=4.93802\times10^{-4}" + UI + r"\;\Longrightarrow\;T_{2S}=2025.10" + U,
    "b4": r"\frac{1}{T_2}=4.93802\times10^{-4}+3.33333\times10^{-4}=8.27135\times10^{-4}" + UI + r"\;\Longrightarrow\;T_2=1208.99" + U,
    "b5": r"A_0=\frac{4\kappa^2}{\xi_0^2(\xi_0^2+\kappa^2)}=0.99995,\qquad T_{2,1}=13.47" + U + r",\ A_1=4.6\times10^{-5}",
    "b6": r"\frac{1}{T_2}\approx\frac{2\times0.005}{20}+\frac{1}{3000}\;\Longrightarrow\;T_2\approx1200.0" + U + r"\quad(-0.74\%)",
    "s1": r"\kappa=\frac{0.005\times5}{2.0}=0.0125,\qquad \xi_0=0.157867",
    "s2": r"\frac{1}{T_{2S}}=\frac{2.0\times0.0249220}{5^2}=1.99376\times10^{-3}" + UI,
    "s3": r"T_{2S}=501.56" + U + r",\qquad T_2=\left(\frac{1}{501.56}+\frac{1}{3000}\right)^{-1}=429.72" + U,
    "s4": r"\frac{1}{T_2}\approx\frac{2\times0.005}{5}+\frac{1}{3000}\;\Longrightarrow\;T_2\approx428.57" + U + r"\quad(-0.27\%)",
    "t1": r"\kappa=\frac{0.005\times0.05}{2.0}=1.25\times10^{-4},\qquad \xi_0=0.0111801",
    "t2": r"\frac{1}{T_{2S}}=\frac{2.0\times1.24995\times10^{-4}}{0.05^2}=0.0999958" + UI,
    "t3": r"T_{2S}=10.0004" + U + r",\qquad T_2=9.9672" + U,
    "t4": r"\frac{1}{T_2}\approx\frac{0.005}{0.05}+\frac{1}{3000}\;\Longrightarrow\;T_2\approx9.9668" + U + r"\quad(-0.004\%)",
    "inv_general": r"\frac{1}{T_2}=\frac{1}{T_{2B}}+\rho_2\frac{S}{V}\;\Longrightarrow\;R=\frac{2\rho_2}{\dfrac{1}{T_2}-\dfrac{1}{T_{2B}}}\ (\text{圆盘}),\qquad h=\frac{\rho_2}{\dfrac{1}{T_2}-\dfrac{1}{T_{2B}}}\ (\text{狭缝})",
    "inv_b": r"R_B=\frac{2\times0.005}{1/1209.26-1/3000}=\frac{0.01}{4.93622\times10^{-4}}=20.26" + UM + r"\quad(+1.3\%)",
    "inv_s": r"R_S=\frac{2\times0.005}{1/429.83-1/3000}=\frac{0.01}{1.99315\times10^{-3}}=5.017" + UM + r"\quad(+0.3\%)",
    "inv_t": r"h=\frac{0.005}{1/10.03-1/3000}=\frac{0.005}{9.94047\times10^{-2}}=0.0503" + UM + r"\quad(+0.6\%)",
    "inv_num":r"R_B=\frac{2\times0.005}{1/1208.99-1/3000}=20.25" + UM + r",\quad R_S=5.016" + UM + r",\quad h=0.0500" + UM,
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
    subprocess.run([XELATEX, "-interaction=nonstopmode", "-halt-on-error", "equations.tex"], cwd=out, check=True,
                   stdout=subprocess.DEVNULL)
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
