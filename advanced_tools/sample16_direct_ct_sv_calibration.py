"""Calibrate Sample 16 on a direct CT V/S axis without a geometry factor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

from advanced_tools.sample16_rho2_cumulative_vc_calibration import (
    DEFAULT_EXPERIMENT,
    read_spectrum,
    weighted_log_quantile,
)
from advanced_tools.sample16_t2_to_equivalent_diameter_fit import DEFAULT_NETWORK_DIR


DEFAULT_TIFF = Path(
    r"E:\python\jupyter\codex\sip模拟\data\ct_backed_samples_raw_copy_20260605"
    r"\sample_16_Grainstone\CT_slices\5-16seged.tiff"
)
DEFAULT_SIMULATION_DIR = Path(
    "simulation_outputs/sample_16_porosity_grouped_nmr_direct_sv_rho2_8p489_20260914"
)


def add_interface_faces(
    counts: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> None:
    transition = mask_a ^ mask_b
    if not np.any(transition):
        return
    owners = np.where(mask_a, labels_a, labels_b)[transition]
    counts += np.bincount(owners, minlength=len(counts))


def direct_ct_pore_sv(
    tiff_path: Path,
    pores: pd.DataFrame,
    voxel_size_um: float,
    pore_value: int,
    block_slices: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    centers_xyz = pores[
        ["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]
    ].to_numpy(float) / (voxel_size_um * 1e-6)
    tree = cKDTree(centers_xyz[:, [2, 1, 0]])
    n_pores = len(pores)
    voxel_counts = np.zeros(n_pores, dtype=np.int64)
    face_counts = np.zeros(n_pores, dtype=np.int64)
    total_voxels = 0

    previous_mask = None
    previous_labels = None
    with tifffile.TiffFile(tiff_path) as tif:
        n_slices = len(tif.pages)
        height, width = tif.pages[0].shape
        for start in range(0, n_slices, block_slices):
            stop = min(start + block_slices, n_slices)
            block = np.stack([tif.pages[z].asarray() for z in range(start, stop)])
            mask = block == pore_value
            coordinates = np.argwhere(mask).astype(float)
            coordinates[:, 0] += start + 0.5
            coordinates[:, 1:] += 0.5
            _, owners = tree.query(coordinates, workers=-1)
            labels = np.full(mask.shape, -1, dtype=np.int32)
            labels[mask] = owners.astype(np.int32)
            voxel_counts += np.bincount(owners, minlength=n_pores)
            total_voxels += len(owners)

            add_interface_faces(
                face_counts, mask[:, :, :-1], mask[:, :, 1:], labels[:, :, :-1], labels[:, :, 1:]
            )
            add_interface_faces(
                face_counts, mask[:, :-1, :], mask[:, 1:, :], labels[:, :-1, :], labels[:, 1:, :]
            )
            if len(mask) > 1:
                add_interface_faces(
                    face_counts, mask[:-1], mask[1:], labels[:-1], labels[1:]
                )
            if previous_mask is not None:
                add_interface_faces(
                    face_counts, previous_mask, mask[0], previous_labels, labels[0]
                )
            previous_mask = mask[-1].copy()
            previous_labels = labels[-1].copy()
            print(f"CT V/S: slices {start + 1}-{stop}/{n_slices}")

    valid = (voxel_counts > 0) & (face_counts > 0)
    result = pd.DataFrame(
        {
            "pore_id": pores["pore_id"].to_numpy(int)[valid],
            "pore_voxel_count": voxel_counts[valid],
            "pore_solid_face_count": face_counts[valid],
            "volume_um3": voxel_counts[valid] * voxel_size_um**3,
            "surface_area_um2": face_counts[valid] * voxel_size_um**2,
            "characteristic_length_v_over_s_um": (
                voxel_size_um * voxel_counts[valid] / face_counts[valid]
            ),
        }
    )
    metadata = {
        "n_slices": n_slices,
        "height": height,
        "width": width,
        "total_voxels": n_slices * height * width,
        "pore_voxels": int(total_voxels),
        "ct_porosity": total_voxels / (n_slices * height * width),
        "physical_pore_solid_faces": int(face_counts.sum()),
        "global_v_over_s_um": voxel_size_um * total_voxels / face_counts.sum(),
        "assigned_pore_regions": int(np.count_nonzero(voxel_counts)),
        "valid_pore_regions": int(np.count_nonzero(valid)),
        "discarded_volume_fraction": float(voxel_counts[~valid].sum() / total_voxels),
    }
    assert int(voxel_counts.sum()) == total_voxels
    return result, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--simulation-dir", type=Path, default=DEFAULT_SIMULATION_DIR)
    parser.add_argument("--voxel-size-um", type=float, default=1.92)
    parser.add_argument("--pore-value", type=int, default=2)
    parser.add_argument("--total-porosity-pct", type=float, default=23.88)
    parser.add_argument("--block-slices", type=int, default=20)
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_outputs/sample_16_direct_ct_sv_calibration_20260914"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.output_dir / "ct_pore_regions_direct_v_over_s.csv"
    geometry_metadata_path = args.output_dir / "ct_sv_geometry_metadata.json"
    if geometry_path.is_file() and geometry_metadata_path.is_file():
        geometry = pd.read_csv(geometry_path)
        geometry_metadata = json.loads(geometry_metadata_path.read_text(encoding="utf-8"))
    else:
        pores = pd.read_csv(args.network_dir / "pores.csv")
        geometry, geometry_metadata = direct_ct_pore_sv(
            args.input_tiff,
            pores,
            args.voxel_size_um,
            args.pore_value,
            args.block_slices,
        )
        geometry.to_csv(geometry_path, index=False)
        geometry_metadata_path.write_text(
            json.dumps(geometry_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    lc = geometry["characteristic_length_v_over_s_um"].to_numpy(float)
    ct_volume = geometry["volume_um3"].to_numpy(float)
    exp_all_t2, exp_all_amp = read_spectrum(args.experiment)
    exp_positive = exp_all_amp > 0
    exp_t2, exp_amp = exp_all_t2[exp_positive], exp_all_amp[exp_positive]

    ct_porosity_pct = 100.0 * geometry_metadata["ct_porosity"]
    resolved_fraction = ct_porosity_pct / args.total_porosity_pct
    unresolved_fraction = 1.0 - resolved_fraction
    raw_ct_quantile = (0.5 - unresolved_fraction) / resolved_fraction
    total_ct_pore_volume_um3 = (
        geometry_metadata["pore_voxels"] * args.voxel_size_um**3
    )
    finite_volume_fraction = float(ct_volume.sum() / total_ct_pore_volume_um3)
    finite_ct_quantile = raw_ct_quantile / finite_volume_fraction
    lc_50 = weighted_log_quantile(lc, ct_volume, finite_ct_quantile)
    t2_50 = weighted_log_quantile(exp_t2, exp_amp, 0.5)
    rho2_sv = lc_50 / (t2_50 * 1e-3)

    sim_manifest = json.loads(
        (args.simulation_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    simulation_rho2 = 1000.0 * float(
        sim_manifest["simulation_params"]["rho_solid_um_per_ms"]
    )
    simulation = pd.read_csv(
        args.simulation_dir / "porosity_group_weighted_t2_inversion_fixed_alpha_476p4.csv"
    )
    sim_t2 = simulation["t2_ms"].to_numpy(float)
    sim_amp = simulation["normalized_amplitude"].to_numpy(float)
    sim_valid = np.isfinite(sim_t2) & np.isfinite(sim_amp) & (sim_t2 > 0) & (sim_amp >= 0)
    sim_t2, sim_amp = sim_t2[sim_valid], sim_amp[sim_valid]

    ct_order = np.argsort(lc)
    ct_lc_sorted = lc[ct_order]
    ct_cdf_raw = np.cumsum(ct_volume[ct_order]) / total_ct_pore_volume_um3
    ct_cdf_corrected = unresolved_fraction + resolved_fraction * ct_cdf_raw
    exp_order = np.argsort(exp_t2)
    exp_lc_cdf = 1e-3 * rho2_sv * exp_t2[exp_order]
    exp_cdf = np.cumsum(exp_amp[exp_order]) / exp_amp.sum()

    display_max_um = 3000.0
    offscale_volume_fraction = float(
        (total_ct_pore_volume_um3 - ct_volume.sum() + ct_volume[lc > display_max_um].sum())
        / total_ct_pore_volume_um3
    )
    edges = np.geomspace(0.8 * lc.min(), display_max_um, args.histogram_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ct_hist, _ = np.histogram(lc, bins=edges, weights=ct_volume)
    ct_hist_norm = ct_hist / ct_hist.max()
    exp_norm = exp_all_amp / exp_all_amp.max()
    sim_norm = sim_amp / sim_amp.max()
    exp_peak_t2 = float(exp_all_t2[np.argmax(exp_all_amp)])
    sim_peak_t2 = float(sim_t2[np.argmax(sim_amp)])
    ct_peak_lc = float(centers[np.argmax(ct_hist)])
    exp_peak_lc = float(1e-3 * rho2_sv * exp_peak_t2)
    sim_peak_lc = float(1e-3 * simulation_rho2 * sim_peak_t2)

    pd.DataFrame(
        {
            "lc_left_um": edges[:-1],
            "lc_right_um": edges[1:],
            "lc_center_um": centers,
            "ct_volume_weighted_frequency": ct_hist,
            "ct_normalized_frequency": ct_hist_norm,
        }
    ).to_csv(args.output_dir / "ct_v_over_s_histogram.csv", index=False)
    pd.DataFrame(
        [
            {
                "ct_porosity_pct": ct_porosity_pct,
                "total_porosity_pct": args.total_porosity_pct,
                "ct_curve_start_vc": unresolved_fraction,
                "raw_ct_quantile_at_adjusted_vc_0p5": raw_ct_quantile,
                "finite_ct_quantile_at_adjusted_vc_0p5": finite_ct_quantile,
                "finite_surface_region_volume_fraction": finite_volume_fraction,
                "ct_v_over_s_at_vc_0p5_um": lc_50,
                "nmr_t2_at_vc_0p5_ms": t2_50,
                "rho2_direct_sv_um_per_s": rho2_sv,
                "ct_histogram_peak_v_over_s_um": ct_peak_lc,
                "experimental_peak_v_over_s_um": exp_peak_lc,
                "simulation_rho2_um_per_s": simulation_rho2,
                "simulation_t2_peak_ms": sim_peak_t2,
                "simulation_peak_v_over_s_um": sim_peak_lc,
                "ct_volume_fraction_above_display_max_or_zero_surface": offscale_volume_fraction,
            }
        ]
    ).to_csv(args.output_dir / "direct_sv_calibration_summary.csv", index=False)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax1.plot(ct_lc_sorted, ct_cdf_raw, color="#8aa0bd", linestyle="--", linewidth=1.4,
             label="CT resolved pore volume (uncorrected)")
    ax1.plot(ct_lc_sorted, ct_cdf_corrected, color="#3f6592", linewidth=2.2,
             label="CT direct $V/S$ + unresolved-volume correction")
    ax1.plot(exp_lc_cdf, exp_cdf, color="#008b78", linewidth=2.5,
             label=rf"experimental NMR ($\rho_2={rho2_sv:.2f}$ $\mu$m/s)")
    ax1.axhline(0.5, color="#555555", linestyle=":", linewidth=0.9)
    ax1.axvline(lc_50, color="#555555", linestyle=":", linewidth=0.9)
    ax1.scatter([lc_50], [0.5], color="#cf4c3c", s=42, zorder=5)
    ax1.set_xscale("log")
    ax1.set_xlim(float(edges[0]), display_max_um)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel(r"Characteristic length, $L_c=V/S$ ($\mu$m)")
    ax1.set_ylabel(r"Cumulative pore-volume fraction, $V_c$")
    ax1.set_title("a  Direct CT $V/S$ calibration")
    ax1.grid(which="major", alpha=0.18)
    ax1.grid(which="minor", axis="x", alpha=0.07)
    ax1.legend(loc="lower right", fontsize=7.4, frameon=True, framealpha=0.92)
    ax1.text(
        0.04,
        0.94,
        rf"$V_c=0.5$: $(V/S)_{{50}}={lc_50:.2f}$ $\mu$m"
        "\n"
        rf"$T_{{2,50}}={t2_50:.0f}$ ms"
        "\n"
        rf"$\rho_2=(V/S)_{{50}}/T_{{2,50}}={rho2_sv:.2f}$ $\mu$m/s",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 4},
    )

    ax2.stairs(ct_hist_norm, edges, color="#4f6f9f", fill=True, alpha=0.13, linewidth=0)
    ax2.stairs(ct_hist_norm, edges, color="#4f6f9f", linewidth=1.9,
               label=rf"CT pore-region $V/S$ (peak {ct_peak_lc:.1f} $\mu$m)")
    ax2.plot(1e-3 * rho2_sv * exp_all_t2, exp_norm, color="#008b78", linewidth=2.5,
             label=rf"experimental NMR $\rightarrow V/S$ (peak {exp_peak_lc:.1f} $\mu$m)")
    ax2.plot(1e-3 * simulation_rho2 * sim_t2, sim_norm, color="#1f2937", linewidth=2.1,
             linestyle="-.", label=rf"grouped 2D NMR, $\rho_2={simulation_rho2:.2f}$ (peak {sim_peak_lc:.1f} $\mu$m)")
    ax2.set_xscale("log")
    ax2.set_xlim(float(edges[0]), display_max_um)
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_xlabel(r"Characteristic length, $L_c=V/S$ ($\mu$m)")
    ax2.set_ylabel("Normalized amplitude / frequency")
    ax2.set_title("b  Geometry-factor-free comparison")
    ax2.grid(which="major", alpha=0.18)
    ax2.grid(which="minor", axis="x", alpha=0.07)
    ax2.text(
        0.98,
        0.95,
        rf"{100.0 * offscale_volume_fraction:.2f}% of CT pore volume has $L_c>{display_max_um:.0f}$ $\mu$m or $S=0$",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7.4,
               frameon=True, framealpha=0.92, ncol=1)
    fig.suptitle("Sample 16: direct CT surface-to-volume calibration", fontsize=11, y=1.01)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    figure_path = args.output_dir / "sample16_direct_ct_v_over_s_calibration.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "method": "Direct CT pore-region V/S calibration at Vc=0.5; no spherical or cylindrical geometry factor.",
        "pore_partition": "Each pore voxel is assigned to the nearest pnextract pore center; only physical pore-solid voxel faces contribute to S; pore-pore partition faces are excluded.",
        "rho2_direct_sv_um_per_s": rho2_sv,
        "simulation_rho2_um_per_s": simulation_rho2,
        "display_max_um": display_max_um,
        "ct_volume_fraction_above_display_max_or_zero_surface": offscale_volume_fraction,
        "geometry_metadata": geometry_metadata,
        "inputs": {
            "ct_tiff": str(args.input_tiff.resolve()),
            "pore_centers": str((args.network_dir / "pores.csv").resolve()),
            "experimental_spectrum": str(args.experiment.resolve()),
            "simulation_dir": str(args.simulation_dir.resolve()),
        },
        "outputs": {
            "figure_png": str(figure_path.resolve()),
            "pore_region_sv_csv": str(geometry_path.resolve()),
            "summary_csv": str((args.output_dir / "direct_sv_calibration_summary.csv").resolve()),
        },
        "limitations": [
            "Pore ownership is a nearest-center voxel partition seeded by pnextract pore centers.",
            "Voxel-face surface area is grid dependent and may overestimate a smooth interface.",
            "The grouped simulation remains a 2D representative-slice approximation, not a full 3D NMR solve.",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
