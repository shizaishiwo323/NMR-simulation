"""Typed result containers for NMR T2 processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class TrimmedSignal:
    """Container describing raw/trimmed views of a single signal.

    The workflow first sorts data by time and then optionally trims the
    pre-peak rising section. Both representations are retained to improve
    traceability of exported artifacts.
    """

    signal_name: str
    raw_time_ms: np.ndarray
    raw_amplitude: np.ndarray
    trimmed_time_ms: np.ndarray
    trimmed_amplitude: np.ndarray
    peak_index_in_sorted_raw: int


@dataclass
class NnlsInversionResult:
    """Result container for fixed-regularization NNLS inversion."""

    signal_name: str
    t2_bins_ms: np.ndarray
    spectrum: np.ndarray
    fit_time_ms: np.ndarray
    fit_amplitude: np.ndarray
    residual: np.ndarray
    regularization: float
    residual_norm: float
    roughness_norm: float
    metadata: Dict[str, Any]


@dataclass
class LCurveInversionResult:
    """Result container for L-curve regularization search and inversion."""

    signal_name: str
    t2_bins_ms: np.ndarray
    spectrum: np.ndarray
    fit_time_ms: np.ndarray
    fit_amplitude: np.ndarray
    residual: np.ndarray
    best_regularization: float
    best_index: int
    alpha_values: np.ndarray
    residual_norms: np.ndarray
    roughness_norms: np.ndarray
    zeta_values: np.ndarray
    eta_values: np.ndarray
    slope_reciprocal_values: np.ndarray
    used_range_filter: bool
    metadata: Dict[str, Any]


@dataclass
class GaussianDecompositionResult:
    """Result container for multi-Gaussian spectrum decomposition."""

    signal_name: str
    peak_count: int
    parameter: np.ndarray
    objective_value: float
    t2_bins_ms: np.ndarray
    original_spectrum: np.ndarray
    fitted_spectrum: np.ndarray
    component_matrix: np.ndarray
    peak_table: pd.DataFrame
    trial_errors: np.ndarray
