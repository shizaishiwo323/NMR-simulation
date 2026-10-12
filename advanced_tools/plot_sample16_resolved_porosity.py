"""Plot Sample 16 volume-weighted resolved porosity versus pore radius."""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(r"E:\python\jupyter\codex\sip模拟\results\sample16_gpu_20260912\pore_network")
PORES_CSV = DATA_DIR / "network_parsed" / "pores.csv"
CT_METADATA = DATA_DIR.parent / "digital_rock" / "metadata.json"
OUTPUT_STEM = DATA_DIR / "sample16_resolved_porosity_vs_pore_radius"


def main() -> None:
    pores = pd.read_csv(PORES_CSV, usecols=["pore_radius_m", "pore_volume_m3"])
    valid = (
        np.isfinite(pores["pore_radius_m"])
        & np.isfinite(pores["pore_volume_m3"])
        & (pores["pore_radius_m"] > 0)
        & (pores["pore_volume_m3"] > 0)
    )
    pores = pores.loc[valid].copy()
    porosity = float(json.loads(CT_METADATA.read_text(encoding="utf-8"))["porosity"])

    grouped = (
        pores.assign(pore_radius_um=pores["pore_radius_m"] * 1e6)
        .groupby("pore_radius_um", as_index=False)["pore_volume_m3"]
        .sum()
        .sort_values("pore_radius_um")
    )
    volumes = grouped["pore_volume_m3"].to_numpy(float)
    grouped["resolved_porosity"] = porosity * np.cumsum(volumes[::-1])[::-1] / volumes.sum()
    grouped.to_csv(f"{OUTPUT_STEM}_source_data.csv", index=False)

    assert len(pores) == 6840
    assert np.isclose(grouped["resolved_porosity"].iloc[0], porosity)
    assert np.all(np.diff(grouped["resolved_porosity"]) <= 0)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4))
    source_x = grouped["pore_radius_um"].to_numpy(float)
    source_y = grouped["resolved_porosity"].to_numpy(float)
    x = np.r_[0.1, source_x, np.nextafter(source_x[-1], np.inf)]
    y = np.r_[porosity, source_y, 0.0]
    marker_targets = np.geomspace(source_x[0], source_x[-1], 36)
    marker_idx = np.unique(np.searchsorted(x, marker_targets).clip(0, len(x) - 1))

    ax.plot(x, y, color="#202020", linewidth=1.05, zorder=2)
    ax.plot(
        x[marker_idx], y[marker_idx], linestyle="none", marker="D", markersize=3.2,
        markerfacecolor="white", markeredgecolor="#202020", markeredgewidth=0.75,
        label="μ-CT / pnextract", zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.1, 300)
    ax.set_ylim(0, 0.205)
    ax.set_yticks(np.arange(0, 0.201, 0.04))
    ax.set_xlabel("Pore radius (μm)")
    ax.set_ylabel(r"Resolved porosity, $\phi_r$")
    ax.set_title("Sample 16", pad=7)
    ax.grid(which="major", color="#8a8a8a", linewidth=0.55, alpha=0.55)
    ax.tick_params(which="both", top=True, right=True, width=0.8)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#333333")
    fig.savefig(f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "input_pores_csv": str(PORES_CSV),
        "input_ct_metadata": str(CT_METADATA),
        "pore_count": int(len(pores)),
        "ct_porosity": porosity,
        "pore_radius_range_um": [float(source_x.min()), float(source_x.max())],
        "weight": "pore_volume_m3",
        "definition": "phi_r(R) = phi_CT * sum(V_i for r_i >= R) / sum(V_i)",
        "output_png": f"{OUTPUT_STEM}.png",
        "output_source_data_csv": f"{OUTPUT_STEM}_source_data.csv",
        "dpi": 600,
    }
    Path(f"{OUTPUT_STEM}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
