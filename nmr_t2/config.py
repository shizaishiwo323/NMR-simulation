"""Configuration models for NMR T2 processing pipelines.

Each dataclass groups parameters for one processing stage so that users can
configure workflows explicitly and reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class NnlsConfig:
    """Configuration for fixed-regularization NNLS inversion.

    Attributes
    ----------
    num_bins:
        Number of logarithmically spaced T2 bins.
    regularization:
        Regularization weight (`eps`) used in the MATLAB-compatible solver.
    t2_min_ms:
        Minimum T2 value in milliseconds.
    t2_max_ms:
        Maximum T2 value in milliseconds.
    min_points_after_trim:
        Minimum point count required after trimming from global peak.
    """

    num_bins: int = 200
    regularization: float = 1.0
    t2_min_ms: float = 1.0
    t2_max_ms: float = 1e4
    min_points_after_trim: int = 10


@dataclass(frozen=True)
class LCurveConfig:
    """Configuration for L-curve-based regularization selection."""

    num_bins: int = 200
    t2_min_ms: float = 1e-2
    t2_max_ms: float = 1e5
    alpha_min: float = 1e-6
    alpha_max: float = 1e2
    alpha_count: int = 60
    slope_reciprocal_target: float = 0.25
    slope_reciprocal_valid_range: Tuple[float, float] = (0.1, 10.0)
    min_points_after_trim: int = 10


@dataclass(frozen=True)
class GaussianConfig:
    """Configuration for Gaussian decomposition of T2 spectra."""

    peak_count: int = 3
    max_function_evals: int = 20000
    max_iterations: int = 2000


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for plotting style and figure export."""

    figure_size_pair: Tuple[float, float] = (14.0, 5.6)
    figure_size_single: Tuple[float, float] = (8.0, 6.0)
    dpi: int = 300
    font_family_candidates: Tuple[str, ...] = ("Helvetica", "Arial", "DejaVu Sans")
    minimum_font_size: int = 16
    axis_title_font_size: float = 18.0
    axis_label_font_size: float = 16.0
    figure_title_font_size: float = 20.0
    tick_label_font_size: float = 16.0
    legend_font_size: float = 16.0
    annotation_font_size: float = 16.0
    raw_scatter_size: float = 14.0
    spectrum_marker_size: float = 12.0
    spectrum_line_width: float = 1.8
    line_colors: Tuple[str, str, str] = field(default_factory=lambda: ("#1f77b4", "#d62728", "#111111"))
