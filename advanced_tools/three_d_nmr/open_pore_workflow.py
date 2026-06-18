"""3D open-pore filtering before slice-based NMR simulation.

The workflow follows the pnextract-style interpretation documented in
``临时.md``: pore voxels are connected in 3D, and only pore components that
touch the selected external boundary faces are treated as open/NMR-responsive.
Closed pore components are retained in the statistics but converted to
non-responsive solid/background for the downstream 2D slice simulations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage

from advanced_tools.batch_tiff_random_slices_nmr import (
    invert_decay,
    save_average_plots,
    save_inversion_outputs,
)
from advanced_tools.png_phase_nmr_decay import SimulationParams, simulate_png
from nmr_t2.config import LCurveConfig, NnlsConfig


DEFAULT_TIFF = Path(
    r"C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\sip模拟"
    r"\data_inventory\ct_backed_samples_raw_copy_20260605\sample_89_Grainstone"
    r"\CT_slices\89seged.tiff"
)


@dataclass
class OpenPoreDetectionResult:
    open_mask: np.ndarray
    component_labels: np.ndarray
    component_voxel_counts: np.ndarray
    open_component_labels: np.ndarray
    total_pore_voxels: int
    open_pore_voxels: int
    closed_pore_voxels: int
    pore_component_count: int
    open_component_count: int
    closed_component_count: int


def connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("connectivity must be one of 6, 18, or 26.")


def boundary_component_labels(labels: np.ndarray) -> np.ndarray:
    faces = [
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ]
    boundary_labels = np.unique(np.concatenate([face.ravel() for face in faces]))
    return boundary_labels[boundary_labels > 0]


def detect_open_pores(
    volume: np.ndarray,
    *,
    pore_value: int = 0,
    solid_value: int = 255,
    connectivity: int = 6,
) -> OpenPoreDetectionResult:
    values = np.unique(volume)
    allowed = {int(pore_value), int(solid_value)}
    unexpected = [int(value) for value in values if int(value) not in allowed]
    if unexpected:
        raise ValueError(f"Unexpected voxel values {unexpected}; expected only {sorted(allowed)}.")

    pore_mask = volume == pore_value
    labels, n_components = ndimage.label(pore_mask, structure=connectivity_structure(connectivity))
    component_counts = np.bincount(labels.ravel())
    open_labels = boundary_component_labels(labels)
    open_mask = np.isin(labels, open_labels)

    total_pore = int(np.sum(pore_mask))
    open_pore = int(np.sum(open_mask))
    closed_pore = total_pore - open_pore
    return OpenPoreDetectionResult(
        open_mask=open_mask,
        component_labels=labels,
        component_voxel_counts=component_counts,
        open_component_labels=open_labels,
        total_pore_voxels=total_pore,
        open_pore_voxels=open_pore,
        closed_pore_voxels=closed_pore,
        pore_component_count=int(n_components),
        open_component_count=int(open_labels.size),
        closed_component_count=int(n_components - open_labels.size),
    )


def write_open_pore_phase_png(open_slice: np.ndarray, output_path: Path) -> None:
    rgb = np.zeros(open_slice.shape + (3,), dtype=np.uint8)
    rgb[:, :] = [255, 255, 0]
    rgb[open_slice] = [255, 0, 0]
    Image.fromarray(rgb).save(output_path)


def save_slice_preview(raw_pore: np.ndarray, open_pore: np.ndarray, output_path: Path) -> None:
    rgb = np.zeros(raw_pore.shape + (3,), dtype=np.uint8)
    rgb[:, :] = [120, 120, 120]
    rgb[raw_pore] = [210, 210, 210]
    rgb[open_pore] = [230, 30, 30]
    Image.fromarray(rgb).save(output_path)


def component_table(result: OpenPoreDetectionResult, voxel_size_um: float) -> pd.DataFrame:
    labels = np.arange(1, result.component_voxel_counts.size, dtype=np.int64)
    counts = result.component_voxel_counts[1:]
    is_open = np.isin(labels, result.open_component_labels)
    voxel_volume_um3 = float(voxel_size_um) ** 3
    return pd.DataFrame(
        {
            "component_label": labels,
            "voxel_count": counts.astype(np.int64),
            "volume_um3": counts.astype(float) * voxel_volume_um3,
            "is_open": is_open,
        }
    )


def parse_slice_indices(values: Sequence[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                out.append(int(item))
    return out


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "original", "full"}:
        return None
    return int(value)


def save_detection_summary(
    result: OpenPoreDetectionResult,
    volume_shape: tuple[int, int, int],
    voxel_size_um: float,
    output_path: Path,
) -> dict[str, float | int | list[int]]:
    total_voxels = int(np.prod(volume_shape))
    voxel_volume_um3 = float(voxel_size_um) ** 3
    row = {
        "z_slices": int(volume_shape[0]),
        "rows": int(volume_shape[1]),
        "cols": int(volume_shape[2]),
        "total_voxels": total_voxels,
        "voxel_size_um": float(voxel_size_um),
        "voxel_volume_um3": voxel_volume_um3,
        "total_pore_voxels": result.total_pore_voxels,
        "open_pore_voxels": result.open_pore_voxels,
        "closed_pore_voxels": result.closed_pore_voxels,
        "total_porosity_fraction": result.total_pore_voxels / max(total_voxels, 1),
        "open_porosity_fraction": result.open_pore_voxels / max(total_voxels, 1),
        "closed_porosity_fraction": result.closed_pore_voxels / max(total_voxels, 1),
        "open_fraction_of_pore_voxels": result.open_pore_voxels / max(result.total_pore_voxels, 1),
        "pore_component_count": result.pore_component_count,
        "open_component_count": result.open_component_count,
        "closed_component_count": result.closed_component_count,
        "open_pore_volume_um3": result.open_pore_voxels * voxel_volume_um3,
        "closed_pore_volume_um3": result.closed_pore_voxels * voxel_volume_um3,
    }
    pd.DataFrame([row]).to_csv(output_path, index=False, encoding="utf-8-sig")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_89_3d_open_pore_nmr"))
    parser.add_argument("--num-slices", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--slice-indices", nargs="*", default=None, help="Optional 0-based slice indices, comma-separated or space-separated.")
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    parser.add_argument("--voxel-size-um", type=float, default=1.70)
    parser.add_argument("--diffusion-um2-per-ms", type=float, default=2.0)
    parser.add_argument("--bulk-t2-ms", type=float, default=3000.0)
    parser.add_argument("--rho-solid-um-per-ms", type=float, default=0.005)
    parser.add_argument("--rho-gas-um-per-ms", type=float, default=0.0)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--t-max-ms", type=float, default=1500.0)
    parser.add_argument("--max-grid-size", type=parse_optional_int, default=256)
    parser.add_argument("--mesh-bulk-size-um", type=float, default=12.0)
    parser.add_argument("--mesh-boundary-size-um", type=float, default=4.0)
    parser.add_argument("--mesh-max-points", type=int, default=50000)
    parser.add_argument("--inversion-mode", choices=["l_curve", "fixed"], default="fixed")
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--alpha-min", type=float, default=1e-6)
    parser.add_argument("--alpha-max", type=float, default=1e2)
    parser.add_argument("--alpha-count", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.input_tiff.exists():
        raise FileNotFoundError(f"Input TIFF not found: {args.input_tiff}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_dir = args.output_dir / "open_pore_phase_png"
    preview_dir = args.output_dir / "slice_previews"
    phase_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 3D TIFF: {args.input_tiff}")
    volume = tifffile.imread(str(args.input_tiff))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF stack, got shape {volume.shape}.")

    print("Detecting 3D open pore components...")
    detection = detect_open_pores(
        volume,
        pore_value=args.pore_value,
        solid_value=args.solid_value,
        connectivity=args.connectivity,
    )
    detection_summary = save_detection_summary(
        detection,
        tuple(int(v) for v in volume.shape),
        args.voxel_size_um,
        args.output_dir / "open_closed_pore_summary.csv",
    )
    component_table(detection, args.voxel_size_um).to_csv(
        args.output_dir / "pore_component_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    explicit_indices = parse_slice_indices(args.slice_indices)
    if explicit_indices is None:
        rng = np.random.default_rng(args.seed)
        selected_indices = sorted(int(i) for i in rng.choice(volume.shape[0], size=args.num_slices, replace=False))
    else:
        selected_indices = sorted(explicit_indices)

    params = SimulationParams(
        pixel_size_x_um=args.voxel_size_um,
        pixel_size_y_um=args.voxel_size_um,
        diffusion_um2_per_ms=args.diffusion_um2_per_ms,
        bulk_t2_ms=args.bulk_t2_ms,
        rho_solid_um_per_ms=args.rho_solid_um_per_ms,
        rho_gas_um_per_ms=args.rho_gas_um_per_ms,
        dt_ms=args.dt_ms,
        t_max_ms=args.t_max_ms,
        max_grid_size=args.max_grid_size,
        solver="triangular",
        mesh_bulk_size_um=args.mesh_bulk_size_um,
        mesh_boundary_size_um=args.mesh_boundary_size_um,
        mesh_max_points=args.mesh_max_points,
    )
    lcurve_cfg = LCurveConfig(
        num_bins=args.t2_bins,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_count=args.alpha_count,
    )
    fixed_cfg = NnlsConfig(
        num_bins=args.t2_bins,
        regularization=args.fixed_alpha,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
    )

    slice_rows: list[dict[str, object]] = []
    mesh_rows: list[dict[str, object]] = []
    normalized_signals: list[np.ndarray] = []
    raw_signals: list[np.ndarray] = []
    time_axis: np.ndarray | None = None
    summaries: list[dict[str, object]] = []

    for sequence_id, slice_index in enumerate(selected_indices, start=1):
        if slice_index < 0 or slice_index >= volume.shape[0]:
            raise ValueError(f"Slice index {slice_index} is outside 0..{volume.shape[0] - 1}.")

        stem = f"sample89_3dopen_slice_{slice_index:04d}"
        raw_pore = volume[slice_index] == args.pore_value
        open_slice = detection.open_mask[slice_index]
        closed_slice = raw_pore & ~open_slice
        phase_png = phase_dir / f"{stem}.png"
        preview_png = preview_dir / f"{stem}_open_closed_preview.png"
        write_open_pore_phase_png(open_slice, phase_png)
        save_slice_preview(raw_pore, open_slice, preview_png)

        row = {
            "sequence_id": sequence_id,
            "slice_index_0_based": slice_index,
            "slice_index_1_based": slice_index + 1,
            "phase_png": str(phase_png.resolve()),
            "preview_png": str(preview_png.resolve()),
            "raw_pore_px": int(np.sum(raw_pore)),
            "open_pore_px": int(np.sum(open_slice)),
            "closed_pore_px": int(np.sum(closed_slice)),
            "raw_2d_porosity_fraction": float(np.mean(raw_pore)),
            "open_2d_porosity_fraction": float(np.mean(open_slice)),
            "closed_2d_porosity_fraction": float(np.mean(closed_slice)),
        }

        if row["open_pore_px"] == 0:
            row["simulation_status"] = "skipped_no_open_pores"
            slice_rows.append(row)
            print(f"[{sequence_id}/{len(selected_indices)}] skipped slice {slice_index}: no open pore pixels")
            continue

        sim_summary = simulate_png(phase_png, args.output_dir, params)
        decay = pd.read_csv(sim_summary["curve_csv"])
        current_time = decay["time_ms"].to_numpy(dtype=float)
        raw_signal = decay["signal"].to_numpy(dtype=float)
        current_signal = decay["normalized_signal"].to_numpy(dtype=float)
        if time_axis is None:
            time_axis = current_time
        elif not np.allclose(time_axis, current_time):
            raise RuntimeError("Time axes differ between slice simulations; cannot compute a direct average.")

        inv_result = invert_decay(current_time, current_signal, stem, args.inversion_mode, lcurve_cfg, fixed_cfg)
        inv_outputs = save_inversion_outputs(inv_result, args.output_dir / stem)
        normalized_signals.append(current_signal)
        raw_signals.append(raw_signal)

        mesh_stats = sim_summary.get("mesh_or_boundary_summary", {})
        row.update(
            {
                "simulation_status": "completed",
                "decay_csv": sim_summary["curve_csv"],
                "decay_png": sim_summary["curve_png"],
                "mesh_png": sim_summary["mesh_png"],
                "pygimli_mesh_bms": sim_summary["pygimli_mesh_bms"],
                "mesh_quality_csv": sim_summary["mesh_quality_csv"],
                "mesh_quality_histogram_png": sim_summary["mesh_quality_histogram_png"],
                **inv_outputs,
            }
        )
        row.update({f"mesh_{key}": value for key, value in mesh_stats.items()})
        mesh_rows.append(
            {
                "sequence_id": sequence_id,
                "slice_index_0_based": slice_index,
                "slice_index_1_based": slice_index + 1,
                "phase_png": str(phase_png.resolve()),
                "mesh_png": sim_summary["mesh_png"],
                "pygimli_mesh_bms": sim_summary["pygimli_mesh_bms"],
                "mesh_quality_csv": sim_summary["mesh_quality_csv"],
                "mesh_quality_histogram_png": sim_summary["mesh_quality_histogram_png"],
                **mesh_stats,
            }
        )
        sim_summary["slice_index_0_based"] = slice_index
        sim_summary["open_closed_slice_counts"] = {
            "raw_pore_px": row["raw_pore_px"],
            "open_pore_px": row["open_pore_px"],
            "closed_pore_px": row["closed_pore_px"],
        }
        sim_summary["inversion_outputs"] = inv_outputs
        summaries.append(sim_summary)
        slice_rows.append(row)
        print(f"[{sequence_id}/{len(selected_indices)}] completed slice {slice_index}")

    if time_axis is None or not raw_signals:
        raise RuntimeError("No selected slices contained open pore pixels for simulation.")

    raw_signal_matrix = np.column_stack(raw_signals)
    raw_initial_sum = float(np.sum(raw_signal_matrix[0, :]))
    volume_weighted_average = np.sum(raw_signal_matrix, axis=1) / max(raw_initial_sum, 1e-30)
    mean_slice_normalized = np.mean(np.column_stack(normalized_signals), axis=1)

    average_result = invert_decay(
        time_axis,
        volume_weighted_average,
        "sample89_3dopen_volume_weighted_average",
        args.inversion_mode,
        lcurve_cfg,
        fixed_cfg,
    )
    average_outputs = save_inversion_outputs(
        average_result,
        args.output_dir / "sample89_3dopen_volume_weighted_average",
    )
    plot_outputs = save_average_plots(
        time_axis,
        normalized_signals,
        volume_weighted_average,
        average_result,
        args.output_dir,
    )

    pd.DataFrame(slice_rows).to_csv(args.output_dir / "selected_slices_and_outputs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mesh_rows).to_csv(args.output_dir / "mesh_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_ms": time_axis, "average_normalized_signal": volume_weighted_average}).to_csv(
        args.output_dir / "average_normalized_decay.csv",
        index=False,
    )
    pd.DataFrame({"time_ms": time_axis, "mean_slice_normalized_signal": mean_slice_normalized}).to_csv(
        args.output_dir / "mean_slice_normalized_decay.csv",
        index=False,
    )
    all_decay = {
        "time_ms": time_axis,
        "volume_weighted_average_normalized_signal": volume_weighted_average,
        "mean_slice_normalized_signal": mean_slice_normalized,
    }
    completed_rows = [row for row in slice_rows if row.get("simulation_status") == "completed"]
    for row, signal in zip(completed_rows, normalized_signals):
        all_decay[f"slice_{int(row['slice_index_0_based']):04d}_normalized_signal"] = signal
    pd.DataFrame(all_decay).to_csv(args.output_dir / "all_slice_normalized_decay_curves.csv", index=False)

    manifest = {
        "input_tiff": str(args.input_tiff.resolve()),
        "source_label_convention": {
            str(args.pore_value): "pore voxel before open/closed filtering",
            str(args.solid_value): "solid matrix",
        },
        "open_pore_definition": (
            "A pore component is open/NMR-responsive only if it is connected in 3D to at least one of the six "
            "external volume faces. Closed components are treated as non-responsive in slice simulations."
        ),
        "connectivity": int(args.connectivity),
        "random_seed": int(args.seed),
        "selected_slice_indices_0_based": [int(i) for i in selected_indices],
        "selected_slice_indices_1_based": [int(i) + 1 for i in selected_indices],
        "detection_summary": detection_summary,
        "simulation_params": params.__dict__,
        "inversion_mode": args.inversion_mode,
        "lcurve_config": lcurve_cfg.__dict__,
        "fixed_inversion_config": fixed_cfg.__dict__,
        "average_signal_mode": "volume_weighted_sum_of_raw_slice_signals_normalized_by_initial_sum",
        "average_outputs": {**average_outputs, **plot_outputs},
        "open_closed_pore_summary_csv": str((args.output_dir / "open_closed_pore_summary.csv").resolve()),
        "pore_component_table_csv": str((args.output_dir / "pore_component_table.csv").resolve()),
        "mesh_summary_csv": str((args.output_dir / "mesh_summary.csv").resolve()),
        "per_slice_summaries": summaries,
        "model_dimension_note": (
            "Open/closed pore connectivity is detected on the full 3D CT volume. The NMR PDE solve remains a "
            "2D triangular-mesh slice ensemble using only 3D-open pore pixels as the responsive domain."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected slices: {selected_indices}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
