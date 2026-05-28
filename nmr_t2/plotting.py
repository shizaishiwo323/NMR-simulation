"""Plotting helpers for NMR T2 workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from .config import PlotConfig
from .models import GaussianDecompositionResult, LCurveInversionResult, NnlsInversionResult


def pick_font_family(candidates: Iterable[str]) -> str:
    """Pick the first installed font from candidates; fallback safely."""

    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


def _enforce_min_font(value: float, minimum: float) -> float:
    """Ensure font size does not drop below configured minimum."""

    return max(float(value), float(minimum), 16.0)


def _apply_common_plot_style(config: PlotConfig) -> None:
    """Apply common plotting style with English-friendly sans fonts."""

    plt.rcParams["font.family"] = pick_font_family(config.font_family_candidates)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def _resolve_font_sizes(config: PlotConfig) -> dict[str, float]:
    """Resolve all font-size categories from configuration with safety floor."""

    return {
        "axis_title": _enforce_min_font(config.axis_title_font_size, config.minimum_font_size),
        "axis_label": _enforce_min_font(config.axis_label_font_size, config.minimum_font_size),
        "figure_title": _enforce_min_font(config.figure_title_font_size, config.minimum_font_size),
        "tick_label": _enforce_min_font(config.tick_label_font_size, config.minimum_font_size),
        "legend": _enforce_min_font(config.legend_font_size, config.minimum_font_size),
        "annotation": _enforce_min_font(config.annotation_font_size, config.minimum_font_size),
    }


def plot_decay_and_spectrum_pair(
    *,
    signal_name: str,
    raw_time_ms: np.ndarray,
    raw_amplitude: np.ndarray,
    t2_bins_ms: np.ndarray,
    spectrum: np.ndarray,
    output_path: Optional[Path] = None,
    config: Optional[PlotConfig] = None,
) -> Optional[Path]:
    """Create a 1x2 figure: raw decay points and T2 spectrum."""

    cfg = config or PlotConfig()
    c_raw, c_line, c_marker = cfg.line_colors

    _apply_common_plot_style(cfg)
    font_sizes = _resolve_font_sizes(cfg)

    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=cfg.figure_size_pair, constrained_layout=True)

    axis_left.scatter(raw_time_ms, raw_amplitude, s=cfg.raw_scatter_size, c=c_raw, alpha=0.85, edgecolors="none")
    axis_left.set_title("Echo decay data", fontsize=font_sizes["axis_title"])
    axis_left.set_xlabel("Time (ms)", fontsize=font_sizes["axis_label"])
    axis_left.set_ylabel("Signal amplitude (a.u.)", fontsize=font_sizes["axis_label"])
    axis_left.tick_params(axis="both", labelsize=font_sizes["tick_label"])

    axis_right.plot(t2_bins_ms, spectrum, color=c_line, linewidth=cfg.spectrum_line_width, alpha=0.9)
    axis_right.scatter(t2_bins_ms, spectrum, s=cfg.spectrum_marker_size, c=c_marker, alpha=0.85)
    axis_right.set_xscale("log")
    axis_right.set_title("T2 spectrum", fontsize=font_sizes["axis_title"])
    axis_right.set_xlabel("T2 (ms)", fontsize=font_sizes["axis_label"])
    axis_right.set_ylabel("Amplitude (a.u.)", fontsize=font_sizes["axis_label"])
    axis_right.tick_params(axis="both", labelsize=font_sizes["tick_label"])

    figure.suptitle(f"Signal: {signal_name}", fontsize=font_sizes["figure_title"])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")

    plt.close(figure)
    return output_path


def plot_lcurve_result(
    result: LCurveInversionResult,
    *,
    output_path: Optional[Path] = None,
    config: Optional[PlotConfig] = None,
) -> Optional[Path]:
    """Plot the L-curve and highlight the selected regularization."""

    cfg = config or PlotConfig()
    c_main, c_best, _ = cfg.line_colors

    _apply_common_plot_style(cfg)
    font_sizes = _resolve_font_sizes(cfg)

    figure, axis = plt.subplots(figsize=cfg.figure_size_single, constrained_layout=True)
    axis.loglog(result.residual_norms, result.roughness_norms, "o-", color=c_main, linewidth=1.6, markersize=4)

    axis.loglog(
        result.residual_norms[result.best_index],
        result.roughness_norms[result.best_index],
        marker="*",
        markersize=14,
        color=c_best,
        linestyle="None",
        label=f"Best eps={result.best_regularization:.3e}",
    )

    axis.set_xlabel("Residual norm ||W(Af - b)||", fontsize=font_sizes["axis_label"])
    axis.set_ylabel("Roughness norm ||Lf||", fontsize=font_sizes["axis_label"])
    axis.set_title(f"L-curve: {result.signal_name}", fontsize=font_sizes["axis_title"])
    axis.tick_params(axis="both", labelsize=font_sizes["tick_label"])
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=font_sizes["legend"], loc="best")

    text = (
        f"Best eps = {result.best_regularization:.3e}\n"
        f"R = {float(result.slope_reciprocal_values[result.best_index]):.4f}"
    )
    axis.text(0.03, 0.05, text, transform=axis.transAxes, fontsize=font_sizes["annotation"])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")

    plt.close(figure)
    return output_path


def plot_gaussian_decomposition(
    result: GaussianDecompositionResult,
    *,
    output_path: Optional[Path] = None,
    config: Optional[PlotConfig] = None,
) -> Optional[Path]:
    """Plot original spectrum, fitted spectrum, and each Gaussian component."""

    cfg = config or PlotConfig()
    _apply_common_plot_style(cfg)
    font_sizes = _resolve_font_sizes(cfg)

    figure, axis = plt.subplots(figsize=cfg.figure_size_single, constrained_layout=True)
    axis.plot(result.t2_bins_ms, result.original_spectrum, color="black", linewidth=2.0, label="Original")
    axis.plot(result.t2_bins_ms, result.fitted_spectrum, color="#d62728", linewidth=2.0, label="Gaussian fit")

    for idx in range(result.component_matrix.shape[0]):
        axis.plot(
            result.t2_bins_ms,
            result.component_matrix[idx, :],
            linewidth=1.2,
            alpha=0.9,
            label=f"Peak {idx + 1}",
        )

    axis.set_xscale("log")
    axis.set_xlabel("T2 (ms)", fontsize=font_sizes["axis_label"])
    axis.set_ylabel("Amplitude (a.u.)", fontsize=font_sizes["axis_label"])
    axis.set_title(f"Gaussian decomposition: {result.signal_name}", fontsize=font_sizes["axis_title"])
    axis.tick_params(axis="both", labelsize=font_sizes["tick_label"])
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=font_sizes["legend"], loc="best")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")

    plt.close(figure)
    return output_path


def plot_nnls_fit_result(
    result: NnlsInversionResult,
    *,
    output_path: Optional[Path] = None,
    config: Optional[PlotConfig] = None,
) -> Optional[Path]:
    """Plot fitted decay curve against resampled inversion time points."""

    cfg = config or PlotConfig()
    _apply_common_plot_style(cfg)
    font_sizes = _resolve_font_sizes(cfg)

    figure, axis = plt.subplots(figsize=cfg.figure_size_single, constrained_layout=True)
    axis.plot(result.fit_time_ms, result.fit_amplitude, color="#d62728", linewidth=2.0, label="NNLS fit")
    axis.set_xlabel("Time (ms)", fontsize=font_sizes["axis_label"])
    axis.set_ylabel("Amplitude (a.u.)", fontsize=font_sizes["axis_label"])
    axis.set_title(f"Fitted decay: {result.signal_name}", fontsize=font_sizes["axis_title"])
    axis.tick_params(axis="both", labelsize=font_sizes["tick_label"])
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=font_sizes["legend"], loc="best")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")

    plt.close(figure)
    return output_path
