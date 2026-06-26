"""Approximate 3D Sample 16 NMR by porosity-grouped representative 2D slices."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import tifffile

from advanced_tools.compare_experimental_t2_with_sim_average import load_experiment_decay
from advanced_tools.sample16_pnextract_t2_overlay import load_node2_pore_table, save_overlay_plot
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_TIFF = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\data_inventory"
    r"\ct_backed_samples_raw_copy_20260605\sample_16_Grainstone"
    r"\CT_slices\1-CTseg\5-16seged.tiff"
)
DEFAULT_EXPERIMENT_DECAY = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\data_inventory"
    r"\ct_backed_samples_raw_copy_20260605\sample_16_Grainstone"
    r"\NMR_original\16-1\T2CPMG\data.csv"
)
DEFAULT_PORE_TABLE = Path("simulation_outputs/sample_16_pnextract_t2_overlay_topaxis/pnextract_pore_radius_table.csv")


@dataclass(frozen=True)
class SliceGroup:
    group_id: int
    start_index: int
    end_index: int
    representative_index: int
    slice_count: int
    porosity_min: float
    porosity_max: float
    porosity_mean: float


def group_porosity_series(porosity: list[float] | np.ndarray, threshold: float) -> list[SliceGroup]:
    values = [float(v) for v in porosity]
    if not values:
        return []
    groups = []
    start = 0
    current_values = [values[0]]
    for index, value in enumerate(values[1:], start=1):
        candidate_values = current_values + [value]
        if max(candidate_values) - min(candidate_values) <= threshold:
            current_values = candidate_values
        else:
            end = index - 1
            groups.append(
                SliceGroup(
                    group_id=len(groups),
                    start_index=start,
                    end_index=end,
                    representative_index=(start + end) // 2,
                    slice_count=end - start + 1,
                    porosity_min=min(current_values),
                    porosity_max=max(current_values),
                    porosity_mean=float(np.mean(current_values)),
                )
            )
            start = index
            current_values = [value]
    end = len(values) - 1
    groups.append(
        SliceGroup(
            group_id=len(groups),
            start_index=start,
            end_index=end,
            representative_index=(start + end) // 2,
            slice_count=end - start + 1,
            porosity_min=min(current_values),
            porosity_max=max(current_values),
            porosity_mean=float(np.mean(current_values)),
        )
    )
    return groups


def candidate_indices_for_group(start_index: int, end_index: int) -> list[int]:
    middle_left = (start_index + end_index) // 2
    candidates = []
    for offset in range(0, end_index - start_index + 1):
        left = middle_left - offset
        right = middle_left + offset + (0 if (end_index - start_index + 1) % 2 else 1)
        if start_index <= left <= end_index and left not in candidates:
            candidates.append(left)
        if start_index <= right <= end_index and right not in candidates:
            candidates.append(right)
    return candidates


def compute_slice_porosity(tiff_path: Path, pore_value: int, solid_value: int) -> pd.DataFrame:
    rows = []
    with tifffile.TiffFile(str(tiff_path)) as tif:
        for index, page in enumerate(tif.pages):
            arr = page.asarray()
            pore = arr == pore_value
            solid = arr == solid_value
            if np.any(~(pore | solid)):
                unexpected = np.unique(arr[~(pore | solid)])
                raise ValueError(f"Unexpected labels in slice {index}: {unexpected[:20].tolist()}")
            rows.append(
                {
                    "slice_index_0_based": index,
                    "slice_index_1_based": index + 1,
                    "pore_px": int(np.sum(pore)),
                    "solid_px": int(np.sum(solid)),
                    "total_px": int(arr.size),
                    "porosity": float(np.sum(pore) / max(arr.size, 1)),
                }
            )
    return pd.DataFrame(rows)


def normalized_spectrum_frame(result) -> pd.DataFrame:
    amplitude = np.asarray(result.spectrum, dtype=float)
    return pd.DataFrame(
        {
            "t2_ms": result.t2_bins_ms,
            "amplitude": amplitude,
            "normalized_amplitude": amplitude / max(float(np.max(amplitude)), 1e-30),
        }
    )


def run_one_slice(args: argparse.Namespace, group: SliceGroup, slice_index: int, group_dir: Path) -> bool:
    command = [
        sys.executable,
        "-m",
        "advanced_tools.batch_tiff_random_slices_nmr",
        "--input-tiff",
        str(args.input_tiff),
        "--output-dir",
        str(group_dir),
        "--sample-name",
        f"sample16_group{group.group_id:03d}",
        "--slice-indices",
        str(slice_index),
        "--pore-value",
        str(args.pore_value),
        "--solid-value",
        str(args.solid_value),
        "--pixel-size-um",
        str(args.pixel_size_um),
        "--diffusion-um2-per-ms",
        str(args.diffusion_um2_per_ms),
        "--bulk-t2-ms",
        str(args.bulk_t2_ms),
        "--rho-solid-um-per-ms",
        str(args.rho_solid_um_per_ms),
        "--rho-gas-um-per-ms",
        str(args.rho_gas_um_per_ms),
        "--dt-ms",
        str(args.dt_ms),
        "--t-max-ms",
        str(args.t_max_ms),
        "--max-grid-size",
        str(args.max_grid_size),
        "--mesh-bulk-size-um",
        str(args.mesh_bulk_size_um),
        "--mesh-boundary-size-um",
        str(args.mesh_boundary_size_um),
        "--mesh-max-points",
        str(args.mesh_max_points),
        "--t2-min-ms",
        str(args.t2_min_ms),
        "--t2-max-ms",
        str(args.t2_max_ms),
        "--t2-bins",
        str(args.t2_bins),
        "--inversion-mode",
        "fixed",
        "--fixed-alpha",
        str(args.fixed_alpha),
    ]
    completed = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (group_dir / "subprocess_stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    return completed.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--experiment-decay", type=Path, default=DEFAULT_EXPERIMENT_DECAY)
    parser.add_argument("--pore-table", type=Path, default=DEFAULT_PORE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_16_porosity_grouped_nmr_topaxis"))
    parser.add_argument("--porosity-threshold", type=float, default=0.01)
    parser.add_argument("--pore-value", type=int, default=2)
    parser.add_argument("--solid-value", type=int, default=1)
    parser.add_argument("--pixel-size-um", type=float, default=1.92)
    parser.add_argument("--diffusion-um2-per-ms", type=float, default=2.0)
    parser.add_argument("--bulk-t2-ms", type=float, default=3000.0)
    parser.add_argument("--rho-solid-um-per-ms", type=float, default=0.005)
    parser.add_argument("--rho-gas-um-per-ms", type=float, default=0.0)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--t-max-ms", type=float, default=1500.0)
    parser.add_argument("--max-grid-size", default="256")
    parser.add_argument("--mesh-bulk-size-um", type=float, default=12.0)
    parser.add_argument("--mesh-boundary-size-um", type=float, default=5.0)
    parser.add_argument("--mesh-max-points", type=int, default=50000)
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    parser.add_argument("--rho-um-per-ms", type=float, default=0.015)
    parser.add_argument("--histogram-bins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_runs_dir = args.output_dir / "group_runs"
    mesh_dir = args.output_dir / "representative_mesh_figures"
    group_runs_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    porosity_frame = compute_slice_porosity(args.input_tiff, args.pore_value, args.solid_value)
    porosity_path = args.output_dir / "slice_porosity_profile.csv"
    porosity_frame.to_csv(porosity_path, index=False)
    groups = group_porosity_series(porosity_frame["porosity"].to_numpy(float), args.porosity_threshold)
    groups_frame = pd.DataFrame([asdict(group) for group in groups])

    signals = []
    rows = []
    mesh_rows = []
    time_axis = None
    for group in groups:
        success = False
        attempts = []
        for candidate in candidate_indices_for_group(group.start_index, group.end_index):
            group_dir = group_runs_dir / f"group_{group.group_id:03d}_slice_{candidate:04d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            attempts.append(candidate)
            if not run_one_slice(args, group, candidate, group_dir):
                continue
            decay = pd.read_csv(group_dir / "average_normalized_decay.csv")
            current_time = decay["time_ms"].to_numpy(dtype=float)
            signal = decay["average_normalized_signal"].to_numpy(dtype=float)
            if time_axis is None:
                time_axis = current_time
            elif not np.allclose(time_axis, current_time):
                raise RuntimeError(f"Time axis differs in group {group.group_id}")
            mesh_png = next(group_dir.glob("*triangular_mesh.png"))
            mesh_dest = mesh_dir / f"group_{group.group_id:03d}_slice_{candidate:04d}_count_{group.slice_count:03d}_mesh.png"
            shutil.copy2(mesh_png, mesh_dest)
            signals.append((group.slice_count, signal))
            rows.append(
                {
                    **asdict(group),
                    "used_representative_index": candidate,
                    "used_representative_1_based": candidate + 1,
                    "weight_slice_count": group.slice_count,
                    "attempted_representatives": ",".join(str(v) for v in attempts),
                    "group_output_dir": str(group_dir.resolve()),
                    "decay_csv": str((group_dir / "average_normalized_decay.csv").resolve()),
                    "mesh_png": str(mesh_dest.resolve()),
                }
            )
            mesh_rows.append(
                {
                    "group_id": group.group_id,
                    "slice_index_0_based": candidate,
                    "slice_index_1_based": candidate + 1,
                    "weight_slice_count": group.slice_count,
                    "mesh_png": str(mesh_dest.resolve()),
                    "source_mesh_png": str(mesh_png.resolve()),
                }
            )
            success = True
            print(f"[{group.group_id + 1}/{len(groups)}] group {group.group_id} used slice {candidate}, count={group.slice_count}")
            break
        if not success:
            raise RuntimeError(f"All representative candidates failed for group {group.group_id}: {attempts}")

    if time_axis is None:
        raise RuntimeError("No grouped signals were simulated.")
    total_weight = float(sum(weight for weight, _ in signals))
    weighted_signal = sum(weight * signal for weight, signal in signals) / total_weight
    weighted_decay_path = args.output_dir / "group_weighted_normalized_decay.csv"
    pd.DataFrame({"time_ms": time_axis, "group_weighted_normalized_signal": weighted_signal}).to_csv(weighted_decay_path, index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "porosity_groups_and_representatives.csv", index=False, encoding="utf-8-sig")
    groups_frame.to_csv(args.output_dir / "porosity_groups_initial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mesh_rows).to_csv(mesh_dir / "representative_mesh_figures_manifest.csv", index=False, encoding="utf-8-sig")

    cfg = NnlsConfig(
        num_bins=args.t2_bins,
        regularization=args.fixed_alpha,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
    )
    exp_time, exp_signal, _ = load_experiment_decay(args.experiment_decay)
    exp_result = invert_single_signal_nnls(exp_time, exp_signal, signal_name="sample16_experiment_fixed", config=cfg)
    sim_result = invert_single_signal_nnls(time_axis, weighted_signal, signal_name="sample16_porosity_group_weighted", config=cfg)
    exp_spectrum = normalized_spectrum_frame(exp_result)
    sim_spectrum = normalized_spectrum_frame(sim_result)
    exp_spectrum_path = args.output_dir / "experiment_t2_inversion_fixed_alpha_476p4.csv"
    sim_spectrum_path = args.output_dir / "porosity_group_weighted_t2_inversion_fixed_alpha_476p4.csv"
    exp_spectrum.to_csv(exp_spectrum_path, index=False)
    sim_spectrum.to_csv(sim_spectrum_path, index=False)

    pores = pd.read_csv(args.pore_table) if args.pore_table.suffix.lower() == ".csv" else load_node2_pore_table(args.pore_table)
    figure_path = args.output_dir / "sample16_porosity_grouped_vs_experiment_t2_topaxis.png"
    hist = save_overlay_plot(
        pores,
        exp_spectrum,
        sim_spectrum,
        figure_path,
        rho_um_per_ms=args.rho_um_per_ms,
        bins=args.histogram_bins,
        histogram_axis_mode="top_pore_diameter",
        xlim_min_ms=args.t2_min_ms,
        xlim_max_ms=args.t2_max_ms,
        top_axis_max_um=3000.0,
    )
    hist_path = args.output_dir / "pnextract_pore_histogram_for_grouped_overlay.csv"
    hist.to_csv(hist_path, index=False)

    manifest = {
        "method": "Contiguous CT slices were grouped so each group has max(porosity)-min(porosity) <= threshold. One middle representative slice was simulated per group; group slice counts weight the averaged decay before fixed-alpha NNLS inversion.",
        "input_tiff": str(args.input_tiff.resolve()),
        "experiment_decay": str(args.experiment_decay.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "porosity_threshold": args.porosity_threshold,
        "num_input_slices": int(len(porosity_frame)),
        "num_groups": int(len(groups)),
        "total_weighted_slices": int(total_weight),
        "simulation_params": {
            "pixel_size_um": args.pixel_size_um,
            "diffusion_um2_per_ms": args.diffusion_um2_per_ms,
            "bulk_t2_ms": args.bulk_t2_ms,
            "rho_solid_um_per_ms": args.rho_solid_um_per_ms,
            "rho_gas_um_per_ms": args.rho_gas_um_per_ms,
            "dt_ms": args.dt_ms,
            "t_max_ms": args.t_max_ms,
            "max_grid_size": args.max_grid_size,
            "mesh_bulk_size_um": args.mesh_bulk_size_um,
            "mesh_boundary_size_um": args.mesh_boundary_size_um,
            "mesh_max_points": args.mesh_max_points,
        },
        "inversion_config": cfg.__dict__,
        "outputs": {
            "slice_porosity_profile_csv": str(porosity_path.resolve()),
            "groups_csv": str((args.output_dir / "porosity_groups_and_representatives.csv").resolve()),
            "weighted_decay_csv": str(weighted_decay_path.resolve()),
            "experiment_spectrum_csv": str(exp_spectrum_path.resolve()),
            "simulation_spectrum_csv": str(sim_spectrum_path.resolve()),
            "overlay_figure_png": str(figure_path.resolve()),
            "mesh_dir": str(mesh_dir.resolve()),
            "mesh_manifest_csv": str((mesh_dir / "representative_mesh_figures_manifest.csv").resolve()),
            "histogram_csv": str(hist_path.resolve()),
        },
        "model_dimension_note": "This is a porosity-grouped 2D representative-slice approximation to the 3D stack, not a full 3D NMR solve.",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Groups: {len(groups)}")
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
