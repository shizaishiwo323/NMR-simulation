"""Compare Sample 16 Avizo and pnextract volume-weighted size distributions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AVIZO_WORKBOOK = Path(
    r"E:\python\jupyter\codex\sip模拟\data\ct_backed_samples_raw_copy_20260605"
    r"\sample_16_Grainstone\CT_slices\avizo\16-avizo-pore throat.xlsx"
)
PNEXTRACT_DIR = (
    ROOT
    / "simulation_outputs/sample_16_pnm_rho2_14p63_raw_volume_nmr_normalized_20260916/network_input"
)
OUTPUT_DIR = ROOT / "simulation_outputs/sample_16_avizo_pnextract_histograms_20260916"
N_BINS = 46


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive(values: pd.Series, weights: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values_array = values.to_numpy(float)
    weights_array = weights.to_numpy(float)
    keep = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (values_array > 0)
        & (weights_array > 0)
    )
    return values_array[keep], weights_array[keep]


def weighted_histogram(
    values: np.ndarray, weights: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=edges, weights=weights)
    return 100.0 * histogram / weights.sum()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    avizo_pores = pd.read_excel(AVIZO_WORKBOOK, sheet_name="Pores")
    avizo_throats = pd.read_excel(AVIZO_WORKBOOK, sheet_name="Throats")
    pn_pores = pd.read_csv(PNEXTRACT_DIR / "pores.csv")
    pn_throats_all = pd.read_csv(PNEXTRACT_DIR / "throats.csv")
    pn_throats = pn_throats_all[
        (pn_throats_all["pore1_id"] > 0) & (pn_throats_all["pore2_id"] > 0)
    ]

    pore_data = {
        "Avizo": positive(2.0 * avizo_pores["EqRadius"], avizo_pores["Volume"]),
        "pnextract": positive(
            2e6 * pn_pores["pore_radius_m"], pn_pores["pore_volume_m3"]
        ),
    }
    throat_data = {
        "Avizo": positive(
            avizo_throats["ChannelLength"],
            avizo_throats["Area"] * avizo_throats["ChannelLength"],
        ),
        "pnextract": positive(
            1e6 * pn_throats["throat_length_m"], pn_throats["throat_volume_m3"]
        ),
    }

    pore_all = np.concatenate([values for values, _ in pore_data.values()])
    throat_all = np.concatenate([values for values, _ in throat_data.values()])
    pore_edges = np.geomspace(pore_all.min(), pore_all.max(), N_BINS + 1)
    throat_edges = np.geomspace(throat_all.min(), throat_all.max(), N_BINS + 1)

    records: list[pd.DataFrame] = []
    histograms: dict[str, dict[str, np.ndarray]] = {"pore": {}, "throat": {}}
    for geometry, data, edges in (
        ("pore", pore_data, pore_edges),
        ("throat", throat_data, throat_edges),
    ):
        for method, (values, weights) in data.items():
            frequency = weighted_histogram(values, weights, edges)
            histograms[geometry][method] = frequency
            records.append(
                pd.DataFrame(
                    {
                        "geometry": geometry,
                        "method": method,
                        "bin_left_um": edges[:-1],
                        "bin_right_um": edges[1:],
                        "bin_center_um": np.sqrt(edges[:-1] * edges[1:]),
                        "volume_weighted_frequency_percent": frequency,
                    }
                )
            )
    source_data = pd.concat(records, ignore_index=True)
    source_data.to_csv(
        OUTPUT_DIR / "sample16_avizo_pnextract_volume_weighted_histograms.csv",
        index=False,
    )

    for geometry in histograms.values():
        for frequency in geometry.values():
            assert np.isclose(frequency.sum(), 100.0)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.9,
        }
    )
    colors = {"Avizo": "#315f68", "pnextract": "#c47a2c"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    panels = (
        (axes[0], "pore", pore_edges, "Pore diameter distribution", r"Pore diameter ($\mu$m)"),
        (axes[1], "throat", throat_edges, "Pore throat length distribution", r"Pore throat length ($\mu$m)"),
    )
    for panel_index, (ax, geometry, edges, title, xlabel) in enumerate(panels):
        for method in ("Avizo", "pnextract"):
            frequency = histograms[geometry][method]
            ax.stairs(
                frequency,
                edges,
                fill=True,
                color=colors[method],
                alpha=0.30,
                linewidth=0,
            )
            ax.stairs(
                frequency,
                edges,
                color=colors[method],
                linewidth=1.25,
                label=method,
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Volume-weighted frequency fraction (%)")
        ax.set_title(title, fontsize=10, pad=7)
        ax.grid(axis="y", color="#b8b8b8", linewidth=0.55, alpha=0.55)
        ax.tick_params(which="both", direction="in", top=True, right=True, length=4)
        ax.text(
            -0.10,
            1.04,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(w_pad=2.2)
    figure_path = OUTPUT_DIR / "sample16_avizo_pnextract_volume_weighted_histograms.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "figure": str(figure_path.resolve()),
        "weighting": {
            "pores": "pore volume",
            "Avizo_throats": "Area * ChannelLength cylindrical-volume proxy",
            "pnextract_throats": "throat_volume_m3",
        },
        "quantities": {
            "panel_a": "pore equivalent diameter",
            "panel_b": "pore throat length",
        },
        "binning": {"scale": "logarithmic", "bins_per_panel": N_BINS, "shared_between_methods": True},
        "records": {
            "Avizo_pores": len(avizo_pores),
            "Avizo_throats": len(avizo_throats),
            "pnextract_pores": len(pn_pores),
            "pnextract_internal_throats": len(pn_throats),
            "pnextract_boundary_throats_excluded": len(pn_throats_all) - len(pn_throats),
        },
        "inputs": {
            "Avizo_workbook_sha256": sha256(AVIZO_WORKBOOK),
            "pnextract_pores_sha256": sha256(PNEXTRACT_DIR / "pores.csv"),
            "pnextract_throats_sha256": sha256(PNEXTRACT_DIR / "throats.csv"),
        },
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
