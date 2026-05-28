"""High-level pipelines that orchestrate NMR T2 processing tasks.

This module is the main entry point for application users. It combines
I/O parsing, inversion, plotting, and exports into consistent workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import GaussianConfig, LCurveConfig, NnlsConfig, PlotConfig
from .gaussian import decompose_spectrum_as_gaussians
from .io_utils import (
    dataframe_columns_to_numeric_xy,
    export_sheet_map_to_excel,
    find_numeric_pair_columns,
    load_decay_table_multi_column,
    read_sheet_map_from_excel,
    safe_token,
    trim_signal_from_global_peak,
)
from .lcurve import invert_single_signal_lcurve
from .nnls import invert_single_signal_nnls
from .plotting import plot_decay_and_spectrum_pair, plot_gaussian_decomposition, plot_lcurve_result


def _build_output_path(output_dir: Path, dataset_name: str, artifact_name: str, suffix: str) -> Path:
    """Build standardized output path as `<dataset>__<artifact>.<suffix>`."""

    safe_dataset = safe_token(dataset_name)
    safe_artifact = safe_token(artifact_name)
    return output_dir / f"{safe_dataset}__{safe_artifact}.{suffix}"


def run_nnls_workbook(
    input_workbook: Path,
    output_dir: Path,
    *,
    config: Optional[NnlsConfig] = None,
    time_to_ms_scale: float = 1.0,
    trim_from_peak: bool = True,
) -> Dict[str, Path]:
    """Run fixed-regularization NNLS inversion for all valid signals in a workbook."""

    cfg = config or NnlsConfig()
    if not input_workbook.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_workbook}")

    dataset_name = input_workbook.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    time_ms, signal_matrix, signal_names, _ = load_decay_table_multi_column(
        input_workbook,
        time_to_ms_scale=float(time_to_ms_scale),
        signal_name_prefix="col",
    )

    spectrum_sheets: Dict[str, pd.DataFrame] = {}
    trimmed_sheets: Dict[str, pd.DataFrame] = {}
    fit_sheets: Dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for idx, signal_name in enumerate(signal_names):
        signal = signal_matrix[:, idx]

        if trim_from_peak:
            trimmed = trim_signal_from_global_peak(
                signal_name,
                time_ms,
                signal,
                min_points_after_trim=int(cfg.min_points_after_trim),
            )
        else:
            trimmed = trim_signal_from_global_peak(
                signal_name,
                time_ms,
                signal,
                min_points_after_trim=2,
            )

        inversion = invert_single_signal_nnls(
            trimmed.trimmed_time_ms,
            trimmed.trimmed_amplitude,
            signal_name=signal_name,
            config=cfg,
        )

        spectrum_sheets[signal_name] = pd.DataFrame(
            {
                "t2_ms": inversion.t2_bins_ms,
                "amplitude": inversion.spectrum,
            }
        )

        trimmed_frame = pd.DataFrame(
            {
                "raw_time_ms": trimmed.raw_time_ms,
                "raw_amplitude": trimmed.raw_amplitude,
                "trimmed_time_ms": np.nan,
                "trimmed_amplitude": np.nan,
            }
        )
        trimmed_frame.loc[: trimmed.trimmed_time_ms.size - 1, "trimmed_time_ms"] = trimmed.trimmed_time_ms
        trimmed_frame.loc[: trimmed.trimmed_amplitude.size - 1, "trimmed_amplitude"] = trimmed.trimmed_amplitude
        trimmed_sheets[signal_name] = trimmed_frame

        fit_sheets[signal_name] = pd.DataFrame(
            {
                "fit_time_ms": inversion.fit_time_ms,
                "fit_amplitude": inversion.fit_amplitude,
                "residual": inversion.residual,
            }
        )

        summary_rows.append(
            {
                "signal_name": signal_name,
                "regularization": inversion.regularization,
                "residual_norm": inversion.residual_norm,
                "roughness_norm": inversion.roughness_norm,
                "raw_points": int(trimmed.raw_time_ms.size),
                "trimmed_points": int(trimmed.trimmed_time_ms.size),
                "peak_index_in_sorted_raw": int(trimmed.peak_index_in_sorted_raw),
            }
        )

    spectrum_path = _build_output_path(output_dir, dataset_name, "nnls_spectrum", "xlsx")
    trimmed_path = _build_output_path(output_dir, dataset_name, "nnls_trimmed_decay", "xlsx")
    fit_path = _build_output_path(output_dir, dataset_name, "nnls_fit", "xlsx")
    summary_csv = _build_output_path(output_dir, dataset_name, "nnls_summary", "csv")
    summary_xlsx = _build_output_path(output_dir, dataset_name, "nnls_summary", "xlsx")

    export_sheet_map_to_excel(spectrum_sheets, spectrum_path)
    export_sheet_map_to_excel(trimmed_sheets, trimmed_path)
    export_sheet_map_to_excel(fit_sheets, fit_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_df.to_excel(summary_xlsx, index=False)

    return {
        "spectrum_xlsx": spectrum_path,
        "trimmed_xlsx": trimmed_path,
        "fit_xlsx": fit_path,
        "summary_csv": summary_csv,
        "summary_xlsx": summary_xlsx,
    }


def run_lcurve_workbook(
    input_workbook: Path,
    output_dir: Path,
    *,
    config: Optional[LCurveConfig] = None,
    plot_config: Optional[PlotConfig] = None,
    time_to_ms_scale: float = 1.0,
    trim_from_peak: bool = True,
) -> Dict[str, Path]:
    """Run L-curve inversion for all valid signals in a workbook."""

    cfg = config or LCurveConfig()
    if not input_workbook.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_workbook}")

    dataset_name = input_workbook.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / f"{safe_token(dataset_name)}__lcurve_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    time_ms, signal_matrix, signal_names, _ = load_decay_table_multi_column(
        input_workbook,
        time_to_ms_scale=float(time_to_ms_scale),
        signal_name_prefix="col",
    )

    spectrum_sheets: Dict[str, pd.DataFrame] = {}
    metrics_sheets: Dict[str, pd.DataFrame] = {}
    trimmed_sheets: Dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for idx, signal_name in enumerate(signal_names):
        signal = signal_matrix[:, idx]

        if trim_from_peak:
            trimmed = trim_signal_from_global_peak(
                signal_name,
                time_ms,
                signal,
                min_points_after_trim=int(cfg.min_points_after_trim),
            )
        else:
            trimmed = trim_signal_from_global_peak(signal_name, time_ms, signal, min_points_after_trim=2)

        result = invert_single_signal_lcurve(
            trimmed.trimmed_time_ms,
            trimmed.trimmed_amplitude,
            signal_name=signal_name,
            config=cfg,
        )

        spectrum_sheets[signal_name] = pd.DataFrame(
            {
                "t2_ms": result.t2_bins_ms,
                "amplitude": result.spectrum,
            }
        )

        metrics_sheets[signal_name] = pd.DataFrame(
            {
                "alpha_regularization": result.alpha_values,
                "zeta": result.zeta_values,
                "eta": result.eta_values,
                "residual_norm": result.residual_norms,
                "roughness_norm": result.roughness_norms,
                "slope_reciprocal": result.slope_reciprocal_values,
                "is_best": np.arange(result.alpha_values.size) == int(result.best_index),
            }
        )

        trimmed_frame = pd.DataFrame(
            {
                "raw_time_ms": trimmed.raw_time_ms,
                "raw_amplitude": trimmed.raw_amplitude,
                "trimmed_time_ms": np.nan,
                "trimmed_amplitude": np.nan,
            }
        )
        trimmed_frame.loc[: trimmed.trimmed_time_ms.size - 1, "trimmed_time_ms"] = trimmed.trimmed_time_ms
        trimmed_frame.loc[: trimmed.trimmed_amplitude.size - 1, "trimmed_amplitude"] = trimmed.trimmed_amplitude
        trimmed_sheets[signal_name] = trimmed_frame

        figure_path = figure_dir / f"{safe_token(signal_name)}__lcurve.png"
        plot_lcurve_result(result, output_path=figure_path, config=plot_config)

        summary_rows.append(
            {
                "signal_name": signal_name,
                "best_regularization": result.best_regularization,
                "best_index": int(result.best_index),
                "best_residual_norm": float(result.residual_norms[result.best_index]),
                "best_roughness_norm": float(result.roughness_norms[result.best_index]),
                "best_slope_reciprocal": float(result.slope_reciprocal_values[result.best_index]),
                "used_range_filter": bool(result.used_range_filter),
                "raw_points": int(trimmed.raw_time_ms.size),
                "trimmed_points": int(trimmed.trimmed_time_ms.size),
                "lcurve_figure": str(figure_path),
            }
        )

    spectrum_path = _build_output_path(output_dir, dataset_name, "lcurve_spectrum", "xlsx")
    metrics_path = _build_output_path(output_dir, dataset_name, "lcurve_metrics", "xlsx")
    trimmed_path = _build_output_path(output_dir, dataset_name, "lcurve_trimmed_decay", "xlsx")
    summary_csv = _build_output_path(output_dir, dataset_name, "lcurve_summary", "csv")
    summary_xlsx = _build_output_path(output_dir, dataset_name, "lcurve_summary", "xlsx")

    export_sheet_map_to_excel(spectrum_sheets, spectrum_path)
    export_sheet_map_to_excel(metrics_sheets, metrics_path)
    export_sheet_map_to_excel(trimmed_sheets, trimmed_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_df.to_excel(summary_xlsx, index=False)

    return {
        "spectrum_xlsx": spectrum_path,
        "metrics_xlsx": metrics_path,
        "trimmed_xlsx": trimmed_path,
        "summary_csv": summary_csv,
        "summary_xlsx": summary_xlsx,
        "figure_dir": figure_dir,
    }


def run_plotting_workbook_pair(
    raw_decay_workbook: Path,
    spectrum_workbook: Path,
    output_dir: Path,
    *,
    plot_config: Optional[PlotConfig] = None,
    time_to_ms_scale: float = 1.0,
) -> Dict[str, Path]:
    """Generate paired decay/T2 figures from raw and spectrum workbooks."""

    if not raw_decay_workbook.exists():
        raise FileNotFoundError(f"Raw decay workbook not found: {raw_decay_workbook}")
    if not spectrum_workbook.exists():
        raise FileNotFoundError(f"Spectrum workbook not found: {spectrum_workbook}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_time_ms, raw_matrix, raw_signal_names, _ = load_decay_table_multi_column(
        raw_decay_workbook,
        time_to_ms_scale=float(time_to_ms_scale),
        signal_name_prefix="col",
    )
    raw_map = {name: raw_matrix[:, idx] for idx, name in enumerate(raw_signal_names)}

    spectrum_map = read_sheet_map_from_excel(spectrum_workbook)
    generated: Dict[str, Path] = {}

    for signal_name, raw_signal in raw_map.items():
        if signal_name not in spectrum_map:
            continue

        spectrum_df = spectrum_map[signal_name]
        x_col, y_col = find_numeric_pair_columns(
            spectrum_df,
            candidate_x=("t2_ms", "time_ms", "time", "t2"),
            candidate_y=("amplitude", "peak", "signal"),
        )
        t2_bins, spectrum = dataframe_columns_to_numeric_xy(spectrum_df, x_col, y_col)

        figure_path = output_dir / f"{safe_token(raw_decay_workbook.stem)}__{safe_token(signal_name)}__decay_t2.png"
        plot_decay_and_spectrum_pair(
            signal_name=signal_name,
            raw_time_ms=raw_time_ms,
            raw_amplitude=raw_signal,
            t2_bins_ms=t2_bins,
            spectrum=spectrum,
            output_path=figure_path,
            config=plot_config,
        )
        generated[signal_name] = figure_path

    return generated


def run_gaussian_decomposition_on_spectrum_workbook(
    spectrum_workbook: Path,
    output_dir: Path,
    *,
    config: Optional[GaussianConfig] = None,
    plot_config: Optional[PlotConfig] = None,
) -> Dict[str, Path]:
    """Run Gaussian decomposition for each spectrum sheet in a workbook."""

    cfg = config or GaussianConfig()
    if not spectrum_workbook.exists():
        raise FileNotFoundError(f"Spectrum workbook not found: {spectrum_workbook}")

    dataset_name = spectrum_workbook.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / f"{safe_token(dataset_name)}__gaussian_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    spectrum_map = read_sheet_map_from_excel(spectrum_workbook)
    component_sheets: Dict[str, pd.DataFrame] = {}
    fit_sheets: Dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for signal_name, spectrum_df in spectrum_map.items():
        x_col, y_col = find_numeric_pair_columns(
            spectrum_df,
            candidate_x=("t2_ms", "time_ms", "time", "t2"),
            candidate_y=("amplitude", "peak", "signal"),
        )
        t2_bins, spectrum = dataframe_columns_to_numeric_xy(spectrum_df, x_col, y_col)

        result = decompose_spectrum_as_gaussians(
            t2_bins,
            spectrum,
            signal_name=signal_name,
            config=cfg,
        )

        fit_frame = pd.DataFrame(
            {
                "t2_ms": result.t2_bins_ms,
                "original_spectrum": result.original_spectrum,
                "fitted_spectrum": result.fitted_spectrum,
            }
        )
        for idx in range(result.component_matrix.shape[0]):
            fit_frame[f"component_{idx + 1}"] = result.component_matrix[idx, :]
        fit_sheets[signal_name] = fit_frame
        component_sheets[signal_name] = result.peak_table

        figure_path = figure_dir / f"{safe_token(signal_name)}__gaussian.png"
        plot_gaussian_decomposition(result, output_path=figure_path, config=plot_config)

        for _, row in result.peak_table.iterrows():
            summary_rows.append(
                {
                    "signal_name": signal_name,
                    "peak_id": int(row["peak_id"]),
                    "height": float(row["height"]),
                    "position_ms": float(row["position_ms"]),
                    "width_log10": float(row["width_log10"]),
                    "area": float(row["area"]),
                    "area_fraction": float(row["area_fraction"]),
                    "objective_value": float(result.objective_value),
                    "gaussian_figure": str(figure_path),
                }
            )

    peak_table_path = _build_output_path(output_dir, dataset_name, "gaussian_peak_table", "xlsx")
    fit_table_path = _build_output_path(output_dir, dataset_name, "gaussian_fit", "xlsx")
    summary_csv = _build_output_path(output_dir, dataset_name, "gaussian_summary", "csv")
    summary_xlsx = _build_output_path(output_dir, dataset_name, "gaussian_summary", "xlsx")

    export_sheet_map_to_excel(component_sheets, peak_table_path)
    export_sheet_map_to_excel(fit_sheets, fit_table_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_df.to_excel(summary_xlsx, index=False)

    return {
        "peak_table_xlsx": peak_table_path,
        "fit_xlsx": fit_table_path,
        "summary_csv": summary_csv,
        "summary_xlsx": summary_xlsx,
        "figure_dir": figure_dir,
    }
