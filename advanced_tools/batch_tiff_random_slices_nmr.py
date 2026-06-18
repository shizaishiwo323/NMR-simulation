"""Run triangular-mesh T2 simulation for random slices from a binary TIFF stack.

Default label convention for the source TIFF:
- 0: pore space, treated as water-filled simulation domain
- 255: solid matrix

The script reuses ``advanced_tools.png_phase_nmr_decay`` for phase-map
classification, triangular meshing, and T2 decay simulation, then applies the
existing L-curve NNLS inversion to each slice and to the average normalized
decay curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from advanced_tools.png_phase_nmr_decay import SimulationParams, simulate_png
from nmr_t2.config import LCurveConfig, NnlsConfig
from nmr_t2.lcurve import invert_single_signal_lcurve
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_TIFF = Path(
    r"C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\sip模拟"
    r"\data_inventory\ct_backed_samples_raw_copy_20260605\sample_89_Grainstone"
    r"\CT_slices\89seged.tiff"
)


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "original", "full"}:
        return None
    return int(value)


def slice_stem(sample_name: str, slice_index: int) -> str:
    return f"{sample_name}_slice_{slice_index:04d}"


def average_signal_name(sample_name: str, num_slices: int) -> str:
    return f"{sample_name}_average_{num_slices}_slices"


def binary_slice_to_phase_png(slice_array: np.ndarray, output_path: Path, pore_value: int, solid_value: int) -> dict[str, int]:
    phase = np.zeros(slice_array.shape + (3,), dtype=np.uint8)
    pore = slice_array == pore_value
    solid = slice_array == solid_value
    phase[pore] = [255, 0, 0]
    phase[solid] = [255, 255, 0]
    if np.any(~(pore | solid)):
        raise ValueError(
            f"Unexpected pixel values in slice: {np.unique(slice_array[~(pore | solid)]).tolist()}. "
            "Only the configured pore and solid values are accepted."
        )
    Image.fromarray(phase).save(output_path)
    return {
        "pore_px": int(np.sum(pore)),
        "solid_px": int(np.sum(solid)),
        "total_px": int(slice_array.size),
    }


def invert_decay(
    time_ms: np.ndarray,
    normalized_signal: np.ndarray,
    signal_name: str,
    mode: str,
    lcurve_cfg: LCurveConfig,
    fixed_cfg: NnlsConfig,
):
    if mode == "fixed":
        return invert_single_signal_nnls(
            time_ms,
            normalized_signal,
            signal_name=signal_name,
            config=fixed_cfg,
        )
    return invert_single_signal_lcurve(
        time_ms,
        normalized_signal,
        signal_name=signal_name,
        config=lcurve_cfg,
    )


def save_inversion_outputs(result, output_prefix: Path) -> dict[str, str | float | int]:
    spectrum_csv = output_prefix.with_name(f"{output_prefix.name}_t2_inversion.csv")
    metrics_csv = output_prefix.with_name(f"{output_prefix.name}_lcurve_metrics.csv")
    fit_csv = output_prefix.with_name(f"{output_prefix.name}_lcurve_fit.csv")
    figure_png = output_prefix.with_name(f"{output_prefix.name}_t2_inversion.png")

    pd.DataFrame(
        {
            "t2_ms": result.t2_bins_ms,
            "amplitude": result.spectrum,
            "normalized_amplitude": result.spectrum / max(float(np.max(result.spectrum)), 1e-30),
        }
    ).to_csv(spectrum_csv, index=False)
    if hasattr(result, "alpha_values"):
        metrics_frame = pd.DataFrame(
            {
                "alpha_regularization": result.alpha_values,
                "residual_norm": result.residual_norms,
                "roughness_norm": result.roughness_norms,
                "zeta": result.zeta_values,
                "eta": result.eta_values,
                "slope_reciprocal": result.slope_reciprocal_values,
                "is_best": np.arange(result.alpha_values.size) == int(result.best_index),
                "inversion_mode": "l_curve",
            }
        )
        best_regularization = float(result.best_regularization)
        best_index = int(result.best_index)
    else:
        metrics_frame = pd.DataFrame(
            {
                "alpha_regularization": [float(result.regularization)],
                "residual_norm": [float(result.residual_norm)],
                "roughness_norm": [float(result.roughness_norm)],
                "zeta": [float(result.residual_norm) ** 2],
                "eta": [float(result.roughness_norm) ** 2],
                "slope_reciprocal": [np.nan],
                "is_best": [True],
                "inversion_mode": ["fixed"],
            }
        )
        best_regularization = float(result.regularization)
        best_index = 0
    metrics_frame.to_csv(metrics_csv, index=False)
    pd.DataFrame(
        {
            "fit_time_ms": result.fit_time_ms,
            "fit_amplitude": result.fit_amplitude,
            "residual": result.residual,
        }
    ).to_csv(fit_csv, index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        result.t2_bins_ms,
        result.spectrum / max(float(np.max(result.spectrum)), 1e-30),
        color="black",
        lw=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("T2 (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(f"{result.signal_name} T2 inversion")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "spectrum_csv": str(spectrum_csv.resolve()),
        "metrics_csv": str(metrics_csv.resolve()),
        "fit_csv": str(fit_csv.resolve()),
        "figure_png": str(figure_png.resolve()),
        "best_regularization": best_regularization,
        "best_index": best_index,
    }


def save_average_plots(
    time_ms: np.ndarray,
    normalized_signals: list[np.ndarray],
    average_signal: np.ndarray,
    average_result,
    output_dir: Path,
) -> dict[str, str]:
    decay_png = output_dir / "average_and_slice_decay_curves.png"
    inversion_png = output_dir / "average_t2_inversion.png"

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for signal in normalized_signals:
        ax.plot(time_ms, signal, color="0.75", lw=0.9)
    ax.plot(time_ms, average_signal, color="black", lw=2.4, label="average")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized NMR signal")
    ax.set_title("10 random slices: normalized T2 decay")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(decay_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        average_result.t2_bins_ms,
        average_result.spectrum / max(float(np.max(average_result.spectrum)), 1e-30),
        color="black",
        lw=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("T2 (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title("Average decay T2 inversion")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(inversion_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "average_decay_png": str(decay_png.resolve()),
        "average_inversion_png": str(inversion_png.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_89_random10_nmr"))
    parser.add_argument("--sample-name", default="sample89")
    parser.add_argument("--num-slices", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument("--pixel-size-um", type=float, default=1.0)
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
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--alpha-min", type=float, default=1e-6)
    parser.add_argument("--alpha-max", type=float, default=1e2)
    parser.add_argument("--alpha-count", type=int, default=60)
    parser.add_argument("--inversion-mode", choices=["l_curve", "fixed"], default="l_curve")
    parser.add_argument("--fixed-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.input_tiff.exists():
        raise FileNotFoundError(f"Input TIFF not found: {args.input_tiff}")
    if args.num_slices <= 0:
        raise ValueError("--num-slices must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_dir = args.output_dir / "phase_png"
    phase_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(str(args.input_tiff)) as tif:
        total_slices = len(tif.pages)
        if args.num_slices > total_slices:
            raise ValueError(f"Requested {args.num_slices} slices but TIFF only contains {total_slices}.")
        rng = np.random.default_rng(args.seed)
        selected_indices = sorted(int(i) for i in rng.choice(total_slices, size=args.num_slices, replace=False))

        params = SimulationParams(
            pixel_size_x_um=args.pixel_size_um,
            pixel_size_y_um=args.pixel_size_um,
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

        slice_rows = []
        mesh_rows = []
        normalized_signals: list[np.ndarray] = []
        time_axis = None
        summaries = []
        for sequence_id, slice_index in enumerate(selected_indices, start=1):
            stem = slice_stem(args.sample_name, slice_index)
            phase_png = phase_dir / f"{stem}.png"
            slice_array = tif.pages[slice_index].asarray()
            counts = binary_slice_to_phase_png(slice_array, phase_png, args.pore_value, args.solid_value)

            sim_summary = simulate_png(phase_png, args.output_dir, params)
            decay = pd.read_csv(sim_summary["curve_csv"])
            current_time = decay["time_ms"].to_numpy(dtype=float)
            current_signal = decay["normalized_signal"].to_numpy(dtype=float)
            if time_axis is None:
                time_axis = current_time
            elif not np.allclose(time_axis, current_time):
                raise RuntimeError("Time axes differ between slice simulations; cannot compute a direct average.")

            inv_result = invert_decay(current_time, current_signal, stem, args.inversion_mode, lcurve_cfg, fixed_cfg)
            inv_outputs = save_inversion_outputs(inv_result, args.output_dir / stem)
            normalized_signals.append(current_signal)

            row = {
                "sequence_id": sequence_id,
                "slice_index_0_based": slice_index,
                "slice_index_1_based": slice_index + 1,
                "phase_png": str(phase_png.resolve()),
                **counts,
                "porosity_2d_fraction": counts["pore_px"] / max(counts["total_px"], 1),
                "decay_csv": sim_summary["curve_csv"],
                "decay_png": sim_summary["curve_png"],
                "mesh_png": sim_summary["mesh_png"],
                "pygimli_mesh_bms": sim_summary["pygimli_mesh_bms"],
                "mesh_quality_csv": sim_summary["mesh_quality_csv"],
                "mesh_quality_histogram_png": sim_summary["mesh_quality_histogram_png"],
                **inv_outputs,
            }
            mesh_stats = sim_summary.get("mesh_or_boundary_summary", {})
            mesh_row = {
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
            row.update({f"mesh_{key}": value for key, value in mesh_stats.items()})
            slice_rows.append(row)
            mesh_rows.append(mesh_row)
            sim_summary["slice_index_0_based"] = slice_index
            sim_summary["phase_pixel_counts_from_tiff"] = counts
            sim_summary["inversion_outputs"] = inv_outputs
            summaries.append(sim_summary)
            print(f"[{sequence_id}/{args.num_slices}] completed slice {slice_index}")

    if time_axis is None:
        raise RuntimeError("No slices were simulated.")

    signal_matrix = np.column_stack(normalized_signals)
    average_signal = np.mean(signal_matrix, axis=1)
    average_name = average_signal_name(args.sample_name, args.num_slices)
    average_result = invert_decay(
        time_axis,
        average_signal,
        average_name,
        args.inversion_mode,
        lcurve_cfg,
        fixed_cfg,
    )
    average_outputs = save_inversion_outputs(average_result, args.output_dir / average_name)
    plot_outputs = save_average_plots(time_axis, normalized_signals, average_signal, average_result, args.output_dir)

    pd.DataFrame(slice_rows).to_csv(args.output_dir / "selected_slices_and_outputs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mesh_rows).to_csv(args.output_dir / "mesh_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_ms": time_axis, "average_normalized_signal": average_signal}).to_csv(
        args.output_dir / "average_normalized_decay.csv",
        index=False,
    )
    all_decay = {"time_ms": time_axis, "average_normalized_signal": average_signal}
    for row, signal in zip(slice_rows, normalized_signals):
        all_decay[f"slice_{row['slice_index_0_based']:04d}_normalized_signal"] = signal
    pd.DataFrame(all_decay).to_csv(args.output_dir / "all_slice_normalized_decay_curves.csv", index=False)

    manifest = {
        "input_tiff": str(args.input_tiff.resolve()),
        "source_label_convention": {
            str(args.pore_value): "pore space treated as water-filled simulation domain",
            str(args.solid_value): "solid matrix",
        },
        "random_seed": int(args.seed),
        "selected_slice_indices_0_based": [int(row["slice_index_0_based"]) for row in slice_rows],
        "selected_slice_indices_1_based": [int(row["slice_index_1_based"]) for row in slice_rows],
        "simulation_params": params.__dict__,
        "inversion_mode": args.inversion_mode,
        "lcurve_config": lcurve_cfg.__dict__,
        "fixed_inversion_config": fixed_cfg.__dict__,
        "average_outputs": {**average_outputs, **plot_outputs},
        "mesh_summary_csv": str((args.output_dir / "mesh_summary.csv").resolve()),
        "per_slice_summaries": summaries,
        "model_dimension_note": (
            "Each selected plane is simulated as an independent 2D slice extracted from a 3D segmented digital rock. "
            "The result is therefore a 10-slice 2D ensemble summary, not a full 3D NMR solve."
        ),
        "scientific_assumptions": [
            "All 0-valued pore pixels are treated as water-filled NMR-active pore space.",
            "255-valued pixels are treated as solid matrix.",
            "No gas phase or separate outside/container label is present in this binary input.",
            "rho_gas is kept at the explicit default 0.0 um/ms; solid-water surface relaxation uses rho_solid.",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected slices: {manifest['selected_slice_indices_0_based']}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
