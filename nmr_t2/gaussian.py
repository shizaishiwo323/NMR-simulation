"""Gaussian peak decomposition for T2 spectra."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from .config import GaussianConfig
from .models import GaussianDecompositionResult


GAUSSIAN_WIDTH_SCALE = 0.60056120439323


def gaussian_component(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Evaluate one Gaussian basis component in log10(T2) space."""

    return np.exp(-((x - center) / (GAUSSIAN_WIDTH_SCALE * width)) ** 2)


def _fit_residual(
    parameter: np.ndarray,
    t2_log10: np.ndarray,
    amplitude: np.ndarray,
    peak_count: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build residual vector for nonlinear optimization.

    For each nonlinear parameter candidate, linear peak heights are solved by
    least squares and constrained to non-negative values via absolute mapping.
    """

    basis = np.zeros((t2_log10.size, peak_count), dtype=float)
    for peak_idx in range(peak_count):
        basis[:, peak_idx] = gaussian_component(t2_log10, parameter[2 * peak_idx], parameter[2 * peak_idx + 1])

    height, *_ = np.linalg.lstsq(basis, amplitude, rcond=None)
    height = np.abs(height)

    model = basis @ height
    residual = model - amplitude
    norm_error = np.linalg.norm(residual)
    return residual, height, norm_error


def _auto_initial_guess(t2_log10: np.ndarray, peak_count: int) -> np.ndarray:
    """Generate automatic initial [center, width, ...] parameters."""

    min_t = float(np.min(t2_log10))
    max_t = float(np.max(t2_log10))
    span = max_t - min_t

    guess = np.zeros(2 * peak_count, dtype=float)
    for idx in range(peak_count):
        k_factor = 0.9 + peak_count - (idx + 1)
        center = min_t + span / k_factor
        width = span / peak_count
        guess[2 * idx : 2 * idx + 2] = [center, width]

    return guess


def decompose_spectrum_as_gaussians(
    t2_bins_ms: np.ndarray,
    spectrum: np.ndarray,
    *,
    signal_name: str,
    config: Optional[GaussianConfig] = None,
    manual_initial_guess: Optional[Sequence[float]] = None,
) -> GaussianDecompositionResult:
    """Decompose one T2 spectrum into multiple Gaussian components."""

    cfg = config or GaussianConfig()

    t2_arr = np.asarray(t2_bins_ms, dtype=float).ravel()
    spectrum_arr = np.asarray(spectrum, dtype=float).ravel()

    valid = np.isfinite(t2_arr) & np.isfinite(spectrum_arr) & (t2_arr > 0)
    t2_arr = t2_arr[valid]
    amp = spectrum_arr[valid]
    if t2_arr.size == 0:
        raise ValueError(f"Signal '{signal_name}' has no valid T2-spectrum points.")

    t2_log = np.log10(t2_arr)
    min_t = float(np.min(t2_log))
    max_t = float(np.max(t2_log))
    span = max_t - min_t

    if manual_initial_guess is None:
        start = _auto_initial_guess(t2_log, int(cfg.peak_count))
    else:
        start = np.asarray(manual_initial_guess, dtype=float).ravel()
        if start.size != 2 * int(cfg.peak_count):
            raise ValueError("Length of `manual_initial_guess` must be 2 * peak_count.")

    lower_bound = np.zeros(2 * int(cfg.peak_count), dtype=float)
    upper_bound = np.zeros(2 * int(cfg.peak_count), dtype=float)
    for idx in range(int(cfg.peak_count)):
        lower_bound[2 * idx] = min_t
        upper_bound[2 * idx] = max_t
        lower_bound[2 * idx + 1] = max(span / 200.0, 1e-4)
        upper_bound[2 * idx + 1] = max(span, 1e-3)

    trial_errors: list[float] = []

    def residual_fn(parameter: np.ndarray) -> np.ndarray:
        residual_vec, _, norm_error = _fit_residual(parameter, t2_log, amp, int(cfg.peak_count))
        trial_errors.append(norm_error)
        return residual_vec

    try:
        lsq_result = least_squares(
            residual_fn,
            x0=start,
            bounds=(lower_bound, upper_bound),
            method="trf",
            max_nfev=int(cfg.max_function_evals),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        parameter = lsq_result.x
        objective_value = float(np.linalg.norm(lsq_result.fun))
    except Exception:

        def scalar_objective(parameter: np.ndarray) -> float:
            residual_vec, _, norm_error = _fit_residual(parameter, t2_log, amp, int(cfg.peak_count))
            trial_errors.append(norm_error)
            return float(np.linalg.norm(residual_vec))

        nm_result = minimize(
            scalar_objective,
            x0=start,
            method="Nelder-Mead",
            options={
                "maxfev": int(cfg.max_function_evals),
                "maxiter": int(cfg.max_iterations),
                "xatol": 1e-8,
                "fatol": 1e-8,
            },
        )
        parameter = nm_result.x
        objective_value = float(nm_result.fun)

    _, height, _ = _fit_residual(parameter, t2_log, amp, int(cfg.peak_count))

    component_matrix = np.zeros((int(cfg.peak_count), t2_log.size), dtype=float)
    fitted = np.zeros(t2_log.size, dtype=float)
    for idx in range(int(cfg.peak_count)):
        component = height[idx] * gaussian_component(t2_log, parameter[2 * idx], parameter[2 * idx + 1])
        component_matrix[idx, :] = component
        fitted += component

    area_each = np.array([np.trapz(component_matrix[idx, :], t2_log) for idx in range(int(cfg.peak_count))], dtype=float)
    total_area = float(np.trapz(fitted, t2_log))
    area_fraction = np.zeros(int(cfg.peak_count), dtype=float) if total_area <= 0 else area_each / total_area

    peak_table = pd.DataFrame(
        {
            "peak_id": np.arange(1, int(cfg.peak_count) + 1, dtype=int),
            "height": height,
            "position_ms": 10**parameter[0::2],
            "width_log10": parameter[1::2],
            "area": area_each,
            "area_fraction": area_fraction,
        }
    ).sort_values("position_ms", ascending=True, kind="mergesort").reset_index(drop=True)

    return GaussianDecompositionResult(
        signal_name=signal_name,
        peak_count=int(cfg.peak_count),
        parameter=parameter,
        objective_value=objective_value,
        t2_bins_ms=t2_arr,
        original_spectrum=amp,
        fitted_spectrum=fitted,
        component_matrix=component_matrix,
        peak_table=peak_table,
        trial_errors=np.asarray(trial_errors, dtype=float),
    )
