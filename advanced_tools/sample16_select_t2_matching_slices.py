"""Select Sample 16 2D slices by fitting simulated T2 spectra to experiment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from advanced_tools.compare_experimental_t2_with_sim_average import load_experiment_decay
from advanced_tools.sample16_pnextract_t2_overlay import load_node2_pore_table, save_overlay_plot
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_EXPERIMENT_DECAY = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\data_inventory"
    r"\ct_backed_samples_raw_copy_20260605\sample_16_Grainstone"
    r"\NMR_original\16-1\T2CPMG\data.csv"
)
DEFAULT_CANDIDATE_DIR = Path("simulation_outputs/sample_16_fitcandidate_slices_nmr_px1p92_b5")
DEFAULT_PORE_TABLE = Path("simulation_outputs/sample_16_pnextract_t2_overlay_topaxis/pnextract_pore_radius_table.csv")


def normalize_nonnegative_weights(weights: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    clean = np.asarray(weights, dtype=float).copy()
    clean[~np.isfinite(clean)] = 0.0
    clean[clean < threshold] = 0.0
    total = float(clean.sum())
    if total <= 0:
        return clean
    return clean / total


def spectrum_fit_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=float)
    target = np.asarray(target, dtype=float)
    if candidate.shape != target.shape:
        raise ValueError("Candidate and target spectra must have the same shape.")
    rmse = float(np.sqrt(np.mean((candidate - target) ** 2)))
    if np.std(candidate) <= 0 or np.std(target) <= 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(candidate, target)[0, 1])
    return {"rmse": rmse, "correlation": correlation}


def build_spectrum_loss_weights(
    t2_ms: np.ndarray,
    short_t2_weight: float = 8.0,
    mid_t2_weight: float = 6.0,
    shoulder_t2_weight: float = 2.5,
) -> np.ndarray:
    t2_ms = np.asarray(t2_ms, dtype=float)
    weights = np.ones_like(t2_ms, dtype=float)
    weights[(t2_ms >= 5.0) & (t2_ms <= 80.0)] = short_t2_weight
    weights[(t2_ms > 80.0) & (t2_ms <= 400.0)] = mid_t2_weight
    weights[(t2_ms > 400.0) & (t2_ms <= 900.0)] = shoulder_t2_weight
    return weights


def weighted_spectrum_fit_metrics(candidate: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if candidate.shape != target.shape or candidate.shape != weights.shape:
        raise ValueError("Candidate, target, and weights must have the same shape.")
    weighted_rmse = float(np.sqrt(np.average((candidate - target) ** 2, weights=weights)))
    return {"weighted_rmse": weighted_rmse}


def combine_weighted_spectra(spectrum_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    combined = np.asarray(spectrum_matrix, dtype=float) @ np.asarray(weights, dtype=float)
    return combined / max(float(np.max(combined)), 1e-30)


def normalized_spectrum_frame(result) -> pd.DataFrame:
    amplitude = np.asarray(result.spectrum, dtype=float)
    return pd.DataFrame(
        {
            "t2_ms": result.t2_bins_ms,
            "amplitude": amplitude,
            "normalized_amplitude": amplitude / max(float(np.max(amplitude)), 1e-30),
        }
    )


def parse_slice_index(column_name: str) -> int:
    match = re.search(r"slice_(\d{4})_normalized_signal", column_name)
    if not match:
        raise ValueError(f"Cannot parse slice index from column: {column_name}")
    return int(match.group(1))


def load_candidate_decays(candidate_dir: Path) -> tuple[np.ndarray, list[tuple[str, int, np.ndarray]]]:
    path = candidate_dir / "all_slice_normalized_decay_curves.csv"
    candidates = []
    if path.exists():
        raw = pd.read_csv(path)
        time_ms = raw["time_ms"].to_numpy(dtype=float)
        for column in raw.columns:
            if column.startswith("slice_") and column.endswith("_normalized_signal"):
                slice_index = parse_slice_index(column)
                candidates.append((f"{candidate_dir.name}:slice_{slice_index:04d}", slice_index, raw[column].to_numpy(dtype=float)))
    else:
        time_ms = None
        for decay_path in sorted(candidate_dir.glob("*slice_*_nmr_decay.csv")):
            match = re.search(r"slice_(\d{4})_nmr_decay", decay_path.name)
            if not match:
                continue
            raw = pd.read_csv(decay_path)
            current_time = raw["time_ms"].to_numpy(dtype=float)
            if time_ms is None:
                time_ms = current_time
            elif not np.allclose(time_ms, current_time):
                raise ValueError(f"Candidate time axis differs in {decay_path}")
            signal_column = "normalized_signal" if "normalized_signal" in raw.columns else raw.columns[1]
            slice_index = int(match.group(1))
            candidates.append(
                (
                    f"{candidate_dir.name}:slice_{slice_index:04d}",
                    slice_index,
                    raw[signal_column].to_numpy(dtype=float),
                )
            )
    if not candidates:
        raise ValueError(f"No candidate slice decay columns found in {path}")
    if time_ms is None:
        raise ValueError(f"No candidate time axis found in {candidate_dir}")
    return time_ms, candidates


def load_candidate_decays_from_dirs(candidate_dirs: list[Path]) -> tuple[np.ndarray, list[tuple[str, int, np.ndarray]]]:
    all_candidates: list[tuple[str, int, np.ndarray]] = []
    time_axis = None
    for candidate_dir in candidate_dirs:
        current_time, current_candidates = load_candidate_decays(candidate_dir)
        if time_axis is None:
            time_axis = current_time
        elif not np.allclose(time_axis, current_time):
            raise ValueError(f"Candidate time axis differs in {candidate_dir}")
        all_candidates.extend(current_candidates)
    if time_axis is None:
        raise ValueError("No candidate directories were provided.")
    return time_axis, all_candidates


def save_spectrum(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-decay", type=Path, default=DEFAULT_EXPERIMENT_DECAY)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--pore-table", type=Path, default=DEFAULT_PORE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_16_t2_matched_slices_topaxis"))
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    parser.add_argument("--rho-um-per-ms", type=float, default=0.015)
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument("--xlim-min-ms", type=float, default=1e-2)
    parser.add_argument("--xlim-max-ms", type=float, default=1e5)
    parser.add_argument("--top-axis-max-um", type=float, default=3000.0)
    parser.add_argument("--combine-mode", choices=["decay", "spectrum"], default="decay")
    parser.add_argument("--loss-weight-short-t2", type=float, default=8.0)
    parser.add_argument("--loss-weight-mid-t2", type=float, default=6.0)
    parser.add_argument("--loss-weight-shoulder-t2", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dirs = args.candidate_dir or [DEFAULT_CANDIDATE_DIR]

    cfg = NnlsConfig(
        num_bins=args.t2_bins,
        regularization=args.fixed_alpha,
        t2_min_ms=args.t2_min_ms,
        t2_max_ms=args.t2_max_ms,
    )
    exp_time, exp_signal, exp_decay = load_experiment_decay(args.experiment_decay)
    exp_result = invert_single_signal_nnls(exp_time, exp_signal, signal_name="sample16_experiment_fixed", config=cfg)
    exp_spectrum = normalized_spectrum_frame(exp_result)
    target = exp_spectrum["normalized_amplitude"].to_numpy(dtype=float)

    sim_time, candidates = load_candidate_decays_from_dirs(candidate_dirs)
    slice_rows = []
    spectra = []
    loss_weights = build_spectrum_loss_weights(
        exp_spectrum["t2_ms"].to_numpy(dtype=float),
        short_t2_weight=args.loss_weight_short_t2,
        mid_t2_weight=args.loss_weight_mid_t2,
        shoulder_t2_weight=args.loss_weight_shoulder_t2,
    )
    for candidate_id, slice_index, signal in candidates:
        result = invert_single_signal_nnls(
            sim_time,
            signal / max(float(signal[0]), 1e-30),
            signal_name=f"{candidate_id}_fixed",
            config=cfg,
        )
        spectrum = normalized_spectrum_frame(result)
        y = spectrum["normalized_amplitude"].to_numpy(dtype=float)
        metrics = spectrum_fit_metrics(y, target)
        weighted_metrics = weighted_spectrum_fit_metrics(y, target, loss_weights)
        slice_rows.append(
            {
                "candidate_id": candidate_id,
                "slice_index_0_based": slice_index,
                "slice_index_1_based": slice_index + 1,
                "peak_t2_ms": float(spectrum.loc[spectrum["normalized_amplitude"].idxmax(), "t2_ms"]),
                **metrics,
                **weighted_metrics,
            }
        )
        spectra.append(y)

    spectrum_matrix = np.column_stack(spectra)
    sqrt_loss_weights = np.sqrt(loss_weights)
    weights = normalize_nonnegative_weights(nnls(spectrum_matrix * sqrt_loss_weights[:, None], target * sqrt_loss_weights)[0])
    if float(weights.sum()) <= 0:
        best_index = int(np.argmin([row["weighted_rmse"] for row in slice_rows]))
        weights[best_index] = 1.0

    weighted_signal = np.zeros_like(sim_time, dtype=float)
    for (_, _, signal), weight in zip(candidates, weights):
        weighted_signal += float(weight) * signal
    weighted_signal = weighted_signal / max(float(weighted_signal[0]), 1e-30)
    if args.combine_mode == "spectrum":
        combined = combine_weighted_spectra(spectrum_matrix, weights)
        weighted_spectrum = pd.DataFrame(
            {
                "t2_ms": exp_spectrum["t2_ms"],
                "amplitude": combined,
                "normalized_amplitude": combined,
            }
        )
    else:
        weighted_result = invert_single_signal_nnls(
            sim_time,
            weighted_signal,
            signal_name="sample16_selected_weighted_slices_fixed",
            config=cfg,
        )
        weighted_spectrum = normalized_spectrum_frame(weighted_result)
    weighted_metrics = spectrum_fit_metrics(weighted_spectrum["normalized_amplitude"].to_numpy(dtype=float), target)
    weighted_metrics.update(
        weighted_spectrum_fit_metrics(weighted_spectrum["normalized_amplitude"].to_numpy(dtype=float), target, loss_weights)
    )

    scores_path = args.output_dir / "candidate_slice_scores.csv"
    weights_path = args.output_dir / "selected_slice_weights.csv"
    exp_spectrum_path = args.output_dir / "experiment_t2_inversion_fixed_alpha_476p4.csv"
    selected_decay_path = args.output_dir / "selected_weighted_decay.csv"
    selected_spectrum_path = args.output_dir / "selected_weighted_t2_inversion_fixed_alpha_476p4.csv"
    figure_path = args.output_dir / "sample16_selected_slices_vs_experiment_t2_topaxis.png"
    manifest_path = args.output_dir / "selection_manifest.json"

    score_frame = pd.DataFrame(slice_rows).sort_values("weighted_rmse")
    score_frame.to_csv(scores_path, index=False)
    weight_frame = pd.DataFrame(
        {
            "slice_index_0_based": [slice_index for _, slice_index, _ in candidates],
            "slice_index_1_based": [slice_index + 1 for _, slice_index, _ in candidates],
            "candidate_id": [candidate_id for candidate_id, _, _ in candidates],
            "weight": weights,
        }
    )
    weight_frame = weight_frame[weight_frame["weight"] > 0].sort_values("weight", ascending=False)
    weight_frame.to_csv(weights_path, index=False)
    save_spectrum(exp_spectrum, exp_spectrum_path)
    pd.DataFrame({"time_ms": sim_time, "selected_weighted_normalized_signal": weighted_signal}).to_csv(
        selected_decay_path,
        index=False,
    )
    save_spectrum(weighted_spectrum, selected_spectrum_path)

    pores = pd.read_csv(args.pore_table) if args.pore_table.suffix.lower() == ".csv" else load_node2_pore_table(args.pore_table)
    hist_frame = save_overlay_plot(
        pores,
        exp_spectrum,
        weighted_spectrum,
        figure_path,
        rho_um_per_ms=args.rho_um_per_ms,
        bins=args.histogram_bins,
        histogram_axis_mode="top_pore_diameter",
        xlim_min_ms=args.xlim_min_ms,
        xlim_max_ms=args.xlim_max_ms,
        top_axis_max_um=args.top_axis_max_um,
    )
    hist_path = args.output_dir / "pnextract_pore_histogram_for_selected_overlay.csv"
    hist_frame.to_csv(hist_path, index=False)

    pnextract_manifest_path = args.pore_table.parent / "run_manifest.json"
    pnextract_provenance = {"run_manifest": str(pnextract_manifest_path.resolve())}
    if pnextract_manifest_path.exists():
        upstream_manifest = json.loads(pnextract_manifest_path.read_text(encoding="utf-8"))
        pnextract_provenance.update(
            {
                "input_tiff": upstream_manifest.get("input_tiff"),
                "pnextract_exe": upstream_manifest.get("pnextract_exe"),
                "node2_path": upstream_manifest.get("node2_path"),
                "label_convention": upstream_manifest.get("label_convention"),
                "voxel_size_um": upstream_manifest.get("voxel_size_um"),
                "display": upstream_manifest.get("display"),
            }
        )

    manifest = {
        "selection_method": "fixed-alpha NNLS inversion of each candidate slice, scored against fixed-alpha experimental inversion; nonnegative slice weights fitted on normalized spectra",
        "combine_mode": args.combine_mode,
        "loss_weights": {
            "short_t2_5_80_ms": args.loss_weight_short_t2,
            "mid_t2_80_400_ms": args.loss_weight_mid_t2,
            "shoulder_t2_400_900_ms": args.loss_weight_shoulder_t2,
        },
        "experiment_decay": str(args.experiment_decay.resolve()),
        "candidate_dirs": [str(candidate_dir.resolve()) for candidate_dir in candidate_dirs],
        "pore_table": str(args.pore_table.resolve()),
        "pnextract_provenance": pnextract_provenance,
        "output_dir": str(args.output_dir.resolve()),
        "inversion_config": cfg.__dict__,
        "selected_slices_0_based": [int(v) for v in weight_frame["slice_index_0_based"].tolist()],
        "selected_slices_1_based": [int(v) for v in weight_frame["slice_index_1_based"].tolist()],
        "selected_weights": [float(v) for v in weight_frame["weight"].tolist()],
        "best_single_candidate_id": str(score_frame.iloc[0]["candidate_id"]),
        "best_single_slice_0_based": int(score_frame.iloc[0]["slice_index_0_based"]),
        "best_single_slice_weighted_rmse": float(score_frame.iloc[0]["weighted_rmse"]),
        "weighted_selection_metrics": weighted_metrics,
        "weighted_selection_peak_t2_ms": float(
            weighted_spectrum.loc[weighted_spectrum["normalized_amplitude"].idxmax(), "t2_ms"]
        ),
        "experiment_peak_t2_ms": float(exp_spectrum.loc[exp_spectrum["normalized_amplitude"].idxmax(), "t2_ms"]),
        "outputs": {
            "candidate_scores_csv": str(scores_path.resolve()),
            "selected_weights_csv": str(weights_path.resolve()),
            "experiment_spectrum_csv": str(exp_spectrum_path.resolve()),
            "selected_decay_csv": str(selected_decay_path.resolve()),
            "selected_spectrum_csv": str(selected_spectrum_path.resolve()),
            "overlay_figure_png": str(figure_path.resolve()),
            "histogram_csv": str(hist_path.resolve()),
        },
        "display": {
            "histogram_axis_mode": "top_pore_diameter",
            "xlim_min_ms": args.xlim_min_ms,
            "xlim_max_ms": args.xlim_max_ms,
            "top_axis_min_um": float(hist_frame["top_axis_min_um"].iloc[0]),
            "top_axis_max_um": float(hist_frame["top_axis_max_um"].iloc[0]),
        },
        "model_dimension_note": "Candidate signals are 2D slice simulations selected to match the experimental T2 spectrum; this is not a full 3D NMR solve.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected slices: {manifest['selected_slices_0_based']}")
    print(f"Weighted RMSE: {weighted_metrics['rmse']:.6f}")
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
