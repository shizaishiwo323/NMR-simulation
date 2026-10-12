"""Common-bin, single-axis porosity overlay; reuse the established radius conversion.

Run: python -m advanced_tools.plot_sample24_common_porosity_overlay --output-dir PATH
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rebin_log_mass(centers, mass, target_edges):
    """Conserve bin mass, assuming uniform density within each original log bin."""
    log_centers = np.log10(centers)
    midpoint = (log_centers[:-1] + log_centers[1:]) / 2
    source_edges = np.r_[2 * log_centers[0] - midpoint[0], midpoint,
                         2 * log_centers[-1] - midpoint[-1]]
    cumulative = np.r_[0., np.cumsum(mass)]
    result = np.diff(np.interp(np.log10(target_edges), source_edges, cumulative))
    assert np.all(result >= -1e-12) and np.isclose(result.sum(), mass.sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    previous = Path("simulation_outputs/sample_24_ct_nmr_calibrated_radius_overlay_20261011")
    manifest_path, source_path = previous / "run_manifest.json", previous / "source_data.csv"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = pd.read_csv(source_path, float_precision="round_trip")
    pore = source[source.series == "CT pore body"]
    throat = source[source.series == "CT internal throat"]
    nmr = source[source.series == "NMR first-test NNLS"]
    # User-specified absolute porosities supersede earlier 29.952% and 10.4706% values.
    nmr_total_pct, ct_total_pct = 29.80, 10.47
    ct_log_edges = np.log10(np.r_[pore.radius_left_um.to_numpy(), pore.radius_right_um.iloc[-1]])
    log_step = float(ct_log_edges[1] - ct_log_edges[0])
    positive = nmr.fraction > 0
    nmr_log_step = float(np.diff(np.log10(nmr.radius_um)).mean())
    lower = np.log10(nmr.loc[positive, "radius_um"].min()) - nmr_log_step / 2
    upper = np.log10(nmr.loc[positive, "radius_um"].max()) + nmr_log_step / 2
    below = max(0, int(np.ceil((ct_log_edges[0] - lower) / log_step)))
    above = max(0, int(np.ceil((upper - ct_log_edges[-1]) / log_step)))
    edges = 10 ** (ct_log_edges[0] + np.arange(-below, len(ct_log_edges) + above) * log_step)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_pct = np.pad(pore.fraction.to_numpy() * ct_total_pct, (below, above))
    throat_pct = np.pad(throat.fraction.to_numpy() * ct_total_pct, (below, above))
    nmr_pct = rebin_log_mass(nmr.radius_um.to_numpy(), nmr.fraction.to_numpy(), edges) * nmr_total_pct
    overlap = np.minimum(pore_pct, throat_pct)
    assert np.isclose(pore_pct.sum() + throat_pct.sum(), ct_total_pct)
    assert np.isclose(nmr_pct.sum(), nmr_total_pct)
    assert np.allclose(np.diff(np.log10(edges)), log_step)

    # Contract: compare absolute volume contributions on identical bins; one descriptive sample.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xscale("log")
    ax.set_xlim(edges[0] / 1.1, edges[-1] * 1.1)
    throat_handle = ax.stairs(throat_pct, edges, fill=True, facecolor="white", edgecolor="black",
                             hatch="////", linewidth=0.8, label=f"CT internal throat ({throat_pct.sum():.2f}%)")
    overlap_handle = ax.stairs(overlap, edges, fill=True, facecolor="#999999", edgecolor="black",
                              linewidth=0.8, label="Histogram overlap")
    pore_handle = ax.stairs(pore_pct, edges, color="black", linewidth=1.1,
                           label=f"CT pore body ({pore_pct.sum():.2f}%)")
    nmr_handle, = ax.plot(centers, nmr_pct, color="black", linewidth=1.6,
                         label=f"NMR first-test NNLS ({nmr_total_pct:.2f}%)")
    ax.set_ylim(0, max(nmr_pct.max(), pore_pct.max(), throat_pct.max()) * 1.2)
    ax.set_xlabel(r"Equivalent pore radius, $r$ ($\mu$m)")
    ax.set_ylabel(r"Porosity per log-radius bin, $\Delta\phi$ (%)")
    radius_per_ms = .003 * previous_manifest["rho2_effective_um_per_s"]
    top = ax.secondary_xaxis("top", functions=(lambda r: r / radius_per_ms, lambda t: t * radius_per_ms))
    top.set_xlabel(r"$T_2$ (ms)", labelpad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[pore_handle, throat_handle, overlap_handle, nmr_handle], loc="upper left",
              ncol=2, fontsize=7, handlelength=2, columnspacing=1.2)
    fig.tight_layout()
    assert not ax.texts and not ax.get_title() and len(fig.axes) == 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_path = args.output_dir / "sample24_ct_nmr_common_porosity.png"
    fig.savefig(image_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = pd.DataFrame({"radius_left_um": edges[:-1], "radius_right_um": edges[1:],
                         "radius_center_um": centers, "ct_pore_porosity_pct": pore_pct,
                         "ct_throat_porosity_pct": throat_pct, "ct_combined_porosity_pct": pore_pct + throat_pct,
                         "nmr_porosity_pct": nmr_pct, "display_overlap_pct": overlap})
    data.to_csv(args.output_dir / "source_data.csv", index=False)
    manifest = {
        "sample": "24", "nmr_test": "first", "nmr_reinverted": False,
        "porosities_source": "User-specified totals: NMR 29.80%, CT 10.47%",
        "nmr_total_porosity_pct": nmr_total_pct, "ct_total_porosity_pct": ct_total_pct,
        "ct_pore_porosity_pct": float(pore_pct.sum()), "ct_throat_porosity_pct": float(throat_pct.sum()),
        "rho2_effective_um_per_s": previous_manifest["rho2_effective_um_per_s"],
        "mapping": previous_manifest["mapping"], "alignment": previous_manifest["alignment"],
        "common_bin_count": len(centers), "common_log10_bin_width": log_step,
        "ct_scaling": "CT volume fractions use a shared pore-body + internal-throat denominator, then multiply by 10.47 percentage points",
        "nmr_scaling": "Sum-normalized first-test NNLS mass conservatively rebinned onto CT-width log-radius bins, then multiply by 29.80 percentage points",
        "rebin_assumption": "Uniform density within each original NMR log bin; integrate cumulative mass at new edges. No reinversion or smoothing",
        "overlap": "Graphic min(pore, throat), not additional volume; do not add it to the CT total",
        "boundary": previous_manifest["boundary"],
        "figure_contract": {"conclusion": "NMR total and CT-resolved volume are compared on one absolute-porosity axis and identical log-radius bins",
                            "archetype": "quantitative grid", "backend": "Python/matplotlib", "export": "600 dpi PNG",
                            "visible_text": "axes, ticks, legend only", "statistics": "one sample, descriptive"},
        "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in (manifest_path, source_path)],
        "output_png": str(image_path.resolve()),
        "checks": {"nmr_sum_pct": float(nmr_pct.sum()), "ct_combined_sum_pct": float((pore_pct + throat_pct).sum()),
                   "single_y_axis": True, "identical_log_bins": True, "no_title_no_annotations": True},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(image_path), "checks": manifest["checks"],
                      "ct_pore_pct": manifest["ct_pore_porosity_pct"], "ct_throat_pct": manifest["ct_throat_porosity_pct"]}, indent=2))


if __name__ == "__main__":
    assert np.allclose(rebin_log_mass(np.array([1., 10.]), np.array([.4, .6]), np.array([.1, 1., 10., 100.])), [.2, .5, .3])
    main()
