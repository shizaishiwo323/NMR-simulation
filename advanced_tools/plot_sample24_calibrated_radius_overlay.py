"""Plot first-test NNLS on the established Vc=1 radius scale, without reinversion.

Run: python -m advanced_tools.plot_sample24_calibrated_radius_overlay --output-dir PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.plot_sample24_ct_nmr_cumulative import CT_DIR, NMR_DIR
from advanced_tools.plot_sample24_nnls_pore_throat_volume_fraction import N_BINS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    calibration_path = Path("simulation_outputs/sample_24_ct_nmr_cumulative_vc1_20261011/run_manifest.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    alignment = calibration["alignments"][0]
    assert alignment["vc"] == 1 and calibration["nmr_test"] == "first"
    rho2 = alignment["rho2_effective_um_per_s"]
    radius_per_ms = 0.003 * rho2
    network = CT_DIR / "pore_network/network_parsed"
    pore_path, throat_path = network / "pores.csv", network / "throats.csv"
    spectrum_path = NMR_DIR / "nnls_spectrum.csv"
    summary_path = NMR_DIR / "nnls_summary.json"
    pores, throats = pd.read_csv(pore_path), pd.read_csv(throat_path)
    internal = (throats.pore1_id > 0) & (throats.pore2_id > 0)
    throat_total, boundary_count = int(internal.sum()), int((~internal).sum())
    throats = throats.loc[internal]
    arrays = []
    for frame, radius_col, volume_col in (
        (pores, "pore_radius_m", "pore_volume_m3"),
        (throats, "throat_radius_m", "throat_volume_m3"),
    ):
        radius, volume = frame[radius_col].to_numpy(float) * 1e6, frame[volume_col].to_numpy(float)
        valid = np.isfinite(radius) & np.isfinite(volume) & (radius > 0) & (volume > 0)
        arrays.append((radius[valid], volume[valid]))
    edges = np.geomspace(min(a[0].min() for a in arrays), max(a[0].max() for a in arrays), N_BINS + 1)
    edges[0] *= 1 - 1e-12
    edges[-1] *= 1 + 1e-12
    denominator = sum(a[1].sum() for a in arrays)
    pore_fraction, throat_fraction = [np.histogram(r, bins=edges, weights=v)[0] / denominator for r, v in arrays]
    overlap = np.minimum(pore_fraction, throat_fraction)
    spectrum = pd.read_csv(spectrum_path).sort_values("t2_ms")
    t2 = spectrum.t2_ms.to_numpy(float)
    signal = spectrum.amplitude_au.to_numpy(float)
    assert np.all(np.isfinite(signal)) and np.all(signal >= 0) and np.all(t2 > 0)
    fraction = signal / signal.sum()
    mapped_radius = radius_per_ms * t2
    positive = signal > 0
    assert np.isclose(mapped_radius[positive].max(), arrays[0][0].max())
    assert np.isclose(fraction.sum(), 1) and np.isclose(pore_fraction.sum() + throat_fraction.sum(), 1)
    assert np.allclose(np.diff(np.log10(t2)), np.diff(np.log10(t2))[0])

    # Figure contract: single-sample descriptive comparison on one conditional radius scale.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax_nmr = ax.twinx()
    ax.set_xscale("log")
    ax.set_xlim(mapped_radius[positive].min() / 1.3, arrays[0][0].max() * 1.25)
    throat_handle = ax.stairs(throat_fraction, edges, fill=True, facecolor="white", edgecolor="black",
                             hatch="////", linewidth=0.8, label="CT internal throat")
    overlap_handle = ax.stairs(overlap, edges, fill=True, facecolor="#999999", edgecolor="black",
                              linewidth=0.8, label="Histogram overlap")
    pore_handle = ax.stairs(pore_fraction, edges, linewidth=1.1, color="black", label="CT pore body")
    nmr_handle, = ax_nmr.plot(mapped_radius, fraction, color="black", linewidth=1.6,
                             label=rf"NMR first-test NNLS ($\rho_2^*={rho2:.2f}$ $\mu$m/s)")
    ax.set_ylim(0, max(pore_fraction.max(), throat_fraction.max()) * 1.2)
    ax_nmr.set_ylim(0, fraction.max() * 1.2)
    ax.set_xlabel(r"Equivalent pore radius, $r$ ($\mu$m)")
    ax.set_ylabel("CT/PNM volume fraction per log-radius bin")
    ax_nmr.set_ylabel("NMR signal fraction per log-radius bin")
    top = ax.secondary_xaxis("top", functions=(lambda r: r / radius_per_ms, lambda t: t * radius_per_ms))
    top.set_xlabel(r"$T_2$ (ms)", labelpad=7)
    ax.spines["top"].set_visible(False)
    ax_nmr.spines["top"].set_visible(False)
    for axis in (ax, ax_nmr, top):
        axis.tick_params(which="both", direction="out", width=0.8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[pore_handle, throat_handle, overlap_handle, nmr_handle], loc="upper left", ncol=2,
              fontsize=7, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout()
    assert all(not a.texts and not a.get_title() for a in (ax, ax_nmr))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    figure = args.output_dir / "sample24_ct_nmr_calibrated_radius_overlay.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    centers = np.sqrt(edges[:-1] * edges[1:])
    source = pd.concat([
        pd.DataFrame({"series": name, "radius_um": centers, "radius_left_um": edges[:-1],
                      "radius_right_um": edges[1:], "fraction": values})
        for name, values in (("CT pore body", pore_fraction), ("CT internal throat", throat_fraction),
                             ("Display histogram overlap", overlap))
    ] + [pd.DataFrame({"series": "NMR first-test NNLS", "radius_um": mapped_radius,
                       "t2_ms": t2, "raw_amplitude_au": signal, "fraction": fraction})], ignore_index=True)
    source.to_csv(args.output_dir / "source_data.csv", index=False)
    paths = [pore_path, throat_path, spectrum_path, summary_path, calibration_path]
    manifest = {
        "sample": "24", "nmr_test": "first", "nmr_reinverted": False,
        "nnls_alpha": json.loads(summary_path.read_text(encoding="utf-8"))["best_alpha"],
        "rho2_effective_um_per_s": rho2, "mapping": "r_um=0.003*rho2_effective*T2_ms; no bulk correction",
        "alignment": "Reuse Vc=1 endpoint calibration; no peak alignment or independent top-axis shift",
        "boundary": "Conditional effective radius; rho2 is not independently measured",
        "ct_counts": {"pores": len(arrays[0][0]), "internal_throats": len(arrays[1][0]),
                      "internal_throats_before_validation": throat_total, "boundary_throats_excluded": boundary_count},
        "ct_histogram": {"bins": N_BINS, "denominator_m3": float(denominator),
                         "normalization": "shared sum of valid pore-body and internal-throat volume",
                         "pore_fraction_sum": float(pore_fraction.sum()), "throat_fraction_sum": float(throat_fraction.sum())},
        "nmr_normalization": "existing nonnegative NNLS amplitude divided by total amplitude; original 200 log bins",
        "ordinate_boundary": "Normalized fractions per bin with separate y axes, not absolute porosity or per-decade densities; CT and NMR log-bin widths differ",
        "overlap": "min(pore_fraction, throat_fraction); graphic overlap only, not physically shared volume",
        "nmr_radius_positive_range_um": [float(mapped_radius[positive].min()), float(mapped_radius[positive].max())],
        "nmr_peak_radius_um": float(mapped_radius[np.argmax(signal)]),
        "figure_contract": {"conclusion": "Compare CT network volume fractions and first-test NMR spectrum on the established radius scale",
                            "archetype": "quantitative grid", "backend": "Python/matplotlib", "export": "600 dpi PNG",
                            "statistics": "one sample; descriptive only", "visible_text": "axes, ticks, legend only"},
        "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "output_png": str(figure.resolve()),
        "qa": {"nmr_sum_one": True, "ct_combined_sum_one": True, "endpoint_matches": True,
               "no_reinversion_no_smoothing": True, "no_title_no_annotation_no_peak_marker": True},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(figure), "rho2": rho2, "nmr_peak_radius_um": manifest["nmr_peak_radius_um"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
