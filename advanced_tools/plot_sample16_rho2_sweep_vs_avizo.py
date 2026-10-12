"""Compare completed Sample 16 grouped-NMR runs with Avizo CT sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("simulation_outputs"))
    parser.add_argument("--avizo-histogram", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NnlsConfig(num_bins=200, regularization=args.alpha, t2_min_ms=0.01, t2_max_ms=100000.0)
    hist = pd.read_csv(args.avizo_histogram)
    edges = np.r_[hist["diameter_left_um"].to_numpy(float), hist["diameter_right_um"].iloc[-1]]
    centers = hist["diameter_center_um"].to_numpy(float)
    pore_volume = hist["pore_volume_weighted_frequency"].to_numpy(float) * 1e18
    throat_volume = hist["throat_volume_weighted_frequency"].to_numpy(float) * 1e18
    pore_peak = float(centers[np.argmax(pore_volume)])
    throat_peak = float(centers[np.argmax(throat_volume)])

    runs = []
    for manifest_path in sorted(args.runs_dir.glob("sample_16_porosity_grouped_nmr*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        decay_path = manifest_path.parent / "group_weighted_normalized_decay.csv"
        if (manifest.get("num_completed_groups") != 70 or manifest.get("total_weighted_slices") != 800
                or manifest.get("skipped_group_ids") or not decay_path.is_file()):
            continue
        rho2 = 1000.0 * float(manifest["simulation_params"]["rho_solid_um_per_ms"])
        runs.append((rho2, manifest_path, decay_path))
    runs.sort(key=lambda row: row[0])
    if not runs or len({rho2 for rho2, _, _ in runs}) != len(runs):
        raise ValueError("No unique, complete Sample 16 grouped-NMR runs found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectra = []
    summaries = []
    for rho2, manifest_path, decay_path in runs:
        decay = pd.read_csv(decay_path)
        time = decay["time_ms"].to_numpy(float)
        signal = decay["group_weighted_normalized_signal"].to_numpy(float)
        if not (len(time) == 301 and np.isclose(signal[0], 1) and np.all(np.diff(signal) <= 1e-8)):
            raise ValueError(f"Invalid grouped decay: {decay_path}")
        result = invert_single_signal_nnls(
            time, signal, signal_name=f"sample16_rho2_{rho2:g}", config=cfg
        )
        amplitude = np.asarray(result.spectrum, dtype=float)
        t2_ms = np.asarray(result.t2_bins_ms, dtype=float)
        diameter_um = 0.006 * rho2 * t2_ms  # sphere: d = 6 rho2 T2, with T2 in ms
        spectrum = pd.DataFrame({
            "rho2_um_per_s": rho2,
            "t2_ms": t2_ms,
            "diameter_um": diameter_um,
            "normalized_amplitude": amplitude / amplitude.max(),
            "source_run": manifest_path.parent.name,
        })
        spectra.append(spectrum)
        peak_index = int(np.argmax(amplitude))
        summaries.append({
            "rho2_um_per_s": rho2,
            "simulated_t2_peak_ms": float(t2_ms[peak_index]),
            "mapped_diameter_peak_um": float(diameter_um[peak_index]),
            "avizo_pore_body_peak_um": pore_peak,
            "avizo_throat_peak_um": throat_peak,
            "group_count": 70,
            "weighted_slice_count": 800,
            "inversion_alpha": args.alpha,
            "source_manifest": str(manifest_path.resolve()),
            "source_decay": str(decay_path.resolve()),
        })

    source_data = pd.concat(spectra, ignore_index=True)
    source_data.to_csv(args.output_dir / "nmr_simulated_spectra_alpha100_mapped_diameter.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "rho2_run_inventory_and_peaks.csv", index=False)
    hist.to_csv(args.output_dir / "avizo_pore_throat_histogram_source.csv", index=False)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(9.1, 5.3))
    nmr_ax = ax.twinx()
    ax.bar(edges[:-1], pore_volume, width=np.diff(edges), align="edge", color="#f4a3a8",
           alpha=0.42, edgecolor="none", label=f"Avizo CT pore body (peak {pore_peak:.0f} µm)")
    ax.bar(edges[:-1], throat_volume, width=np.diff(edges), align="edge", color="none",
           edgecolor="#d98689", linewidth=0.7, alpha=0.72,
           label=f"Avizo CT throat (peak {throat_peak:.0f} µm)")
    colors = plt.cm.cividis(np.linspace(0.12, 0.92, len(runs)))
    for spectrum, (rho2, _, _), color in zip(spectra, runs, colors):
        peak = spectrum.loc[spectrum["normalized_amplitude"].idxmax(), "diameter_um"]
        nmr_ax.plot(spectrum["diameter_um"], spectrum["normalized_amplitude"],
                    linewidth=1.65, color=color,
                    label=rf"$\rho_2={rho2:.3f}$ $\mu$m/s (peak {peak:.0f} $\mu$m)")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 1500)
    ax.set_ylim(bottom=0)
    nmr_ax.set_ylim(0, 1.05)
    ax.set_xlabel("Equivalent pore-body diameter, d (µm)")
    ax.set_ylabel("Avizo CT volume per bin (µm³)", color="#d95259")
    nmr_ax.set_ylabel("Normalized simulated NMR amplitude")
    ax.tick_params(axis="y", colors="#d95259")
    ax.spines["left"].set_color("#d95259")
    ax.set_title("Sample 16: Avizo CT vs grouped-2D NMR surface-relaxivity runs")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = nmr_ax.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=7.6,
              frameon=False, ncol=2, columnspacing=1.1, handlelength=2.0)
    fig.tight_layout()
    figure_path = args.output_dir / "sample16_nmr_rho2_sweep_vs_avizo_ct.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps({
        "figure": str(figure_path.resolve()),
        "method": "Completed porosity-grouped 2D representative-slice simulations; 70 groups and 800 slices per run. Existing weighted decays reinverted with common fixed alpha. Each T2 spectrum mapped to sphere-equivalent diameter using its own simulated rho2.",
        "inversion_alpha": args.alpha,
        "rho2_values_um_per_s": [rho2 for rho2, _, _ in runs],
        "avizo_histogram": str(args.avizo_histogram.resolve()),
        "geometry_note": "NMR maps pore-body equivalent diameter; Avizo throats are context only. Throat volume is the source Avizo area-times-channel-length proxy.",
        "identifiability_note": "Comparison with CT after using each run's rho2 to map T2 to diameter cannot alone identify the physical surface relaxivity; independent experimental T2 comparison is needed.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Runs: {len(runs)}")
    print(pd.DataFrame(summaries)[["rho2_um_per_s", "simulated_t2_peak_ms", "mapped_diameter_peak_um"]].to_string(index=False))
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
