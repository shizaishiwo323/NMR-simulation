"""Compare Sample 16 Vc=0.5 and 200-um peak calibration schemes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.sample16_t2_to_equivalent_diameter_fit import (
    DEFAULT_NETWORK_DIR,
    positive_arrays,
)
from advanced_tools.sample16_t2_to_geometry_diameter_fixed_rho import (
    t2_to_diameter_um,
)


OUTPUT_DIR = Path("simulation_outputs/sample_16_vc0p5_vs_peak200_comparison_20260914")
SCHEMES = (
    {
        "name": "Paper criterion: cumulative median",
        "rho2": 13.498019729684174,
        "vc": 0.5,
        "source": Path(
            "simulation_outputs/sample_16_paper_vc0p5_rho2_13p498_exp500_sim100_20260914"
        ),
    },
    {
        "name": r"Peak criterion: $d_{exp,peak}=200$ $\mu$m",
        "rho2": 20.741902789100774,
        "vc": 0.6288314281920093,
        "source": Path(
            "simulation_outputs/sample_16_peak200_rho2_20p742_exp500_sim100_20260914"
        ),
    },
)


def normalized_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    t2 = frame["t2_ms"].to_numpy(float)
    amplitude = frame["amplitude"].to_numpy(float)
    return t2, amplitude / max(float(amplitude.max()), 1e-30)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pores = pd.read_csv(DEFAULT_NETWORK_DIR / "pores.csv")
    throats = pd.read_csv(DEFAULT_NETWORK_DIR / "throats.csv")
    pore_diameter, pore_volume = positive_arrays(
        2e6 * pores["pore_radius_m"], pores["pore_volume_m3"]
    )
    throat_diameter, throat_volume = positive_arrays(
        2e6 * throats["throat_radius_m"], throats["throat_volume_m3"]
    )
    edges = np.geomspace(
        0.8 * min(float(pore_diameter.min()), float(throat_diameter.min())),
        1.2 * max(float(pore_diameter.max()), float(throat_diameter.max())),
        81,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_hist, _ = np.histogram(pore_diameter, bins=edges, weights=pore_volume)
    throat_hist, _ = np.histogram(throat_diameter, bins=edges, weights=throat_volume)
    pore_hist = pore_hist / pore_hist.max()
    throat_hist = throat_hist / throat_hist.max()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharex=True, sharey=True)
    rows = []
    for index, (ax, scheme) in enumerate(zip(axes, SCHEMES, strict=True)):
        exp_t2, exp_amp = normalized_spectrum(
            scheme["source"] / "experiment_t2_inversion_alpha_500.csv"
        )
        sim_t2, sim_amp = normalized_spectrum(
            scheme["source"] / "simulation_t2_inversion_alpha_100.csv"
        )
        rho2 = float(scheme["rho2"])
        exp_diameter = t2_to_diameter_um(exp_t2, rho2, 3.0)
        sim_diameter = t2_to_diameter_um(sim_t2, rho2, 3.0)
        exp_peak = float(exp_diameter[np.argmax(exp_amp)])
        sim_peak = float(sim_diameter[np.argmax(sim_amp)])
        vc_text = (
            rf"criterion $V_c={scheme['vc']:.3f}$"
            if index == 0
            else rf"curve crossing $V_c={scheme['vc']:.3f}$"
        )
        rows.append(
            {
                "scheme": scheme["name"],
                "rho2_um_per_s": rho2,
                "reference_vc": scheme["vc"],
                "experimental_peak_diameter_um": exp_peak,
                "simulation_peak_diameter_um": sim_peak,
            }
        )

        ax.stairs(pore_hist, edges, color="#4f6f9f", fill=True, alpha=0.12, linewidth=0)
        ax.stairs(pore_hist, edges, color="#4f6f9f", linewidth=1.8, label="PNM pore body")
        ax.stairs(
            throat_hist,
            edges,
            color="#c45d16",
            linewidth=1.7,
            linestyle="--",
            label="PNM throat",
        )
        ax.plot(exp_diameter, exp_amp, color="#008577", linewidth=2.4, label="experimental NMR")
        ax.plot(
            sim_diameter,
            sim_amp,
            color="#1f2937",
            linewidth=2.1,
            linestyle="-.",
            label="grouped 2D NMR",
        )
        ax.set_xscale("log")
        ax.set_xlim(0.02, 1500)
        ax.set_ylim(-0.02, 1.08)
        ax.set_title(f"{'ab'[index]}  {scheme['name']}")
        ax.set_xlabel(r"Sphere-equivalent diameter, $d=6\rho_2T_2$ ($\mu$m)")
        ax.grid(which="major", alpha=0.18)
        ax.grid(which="minor", axis="x", alpha=0.07)
        ax.text(
            0.04,
            0.95,
            rf"$\rho_2={rho2:.2f}$ $\mu$m/s"
            + "\n"
            + vc_text
            + "\n"
            + rf"peaks: exp {exp_peak:.0f}, sim {sim_peak:.0f} $\mu$m",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 4},
        )

    axes[0].set_ylabel("Normalized amplitude / frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, fontsize=8)
    fig.suptitle(
        r"Sample 16: $V_c=0.5$ calibration versus 200-$\mu$m peak alignment"
        "\n"
        r"experimental $\alpha_{inv}=500$; simulation $\alpha_{inv}=100$",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    figure = OUTPUT_DIR / "sample16_vc0p5_vs_peak200_comparison.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = pd.DataFrame(rows)
    assert np.isclose(summary.loc[1, "experimental_peak_diameter_um"], 200.0)
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    manifest = {
        "method": "Side-by-side comparison of the paper Vc=0.5 criterion and experimental peak alignment.",
        "geometry": "sphere-equivalent diameter d=6*rho2*T2",
        "experimental_regularization": 500,
        "simulation_regularization": 100,
        "schemes": rows,
        "figure": str(figure.resolve()),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
