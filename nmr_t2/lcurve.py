"""L-curve-driven regularization selection for T2 NNLS inversion."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import LCurveConfig
from .models import LCurveInversionResult
from .nnls import estimate_noise_vector, solve_t2_nnls_matlab_compatible


def select_best_alpha_by_reciprocal_slope(
    reciprocal_slope_values: np.ndarray,
    *,
    target: float,
    valid_range: Tuple[float, float],
) -> Tuple[int, bool]:
    """Select best index by closest reciprocal slope to target.

    Selection priority:
    1) finite values inside `valid_range`
    2) otherwise any finite value
    3) otherwise fallback to middle index
    """

    values = np.asarray(reciprocal_slope_values, dtype=float)
    if values.size == 0:
        raise ValueError("Empty reciprocal slope array.")

    finite = np.isfinite(values)
    lo, hi = float(valid_range[0]), float(valid_range[1])
    in_range = finite & (values >= lo) & (values <= hi)

    if np.any(in_range):
        candidates = np.where(in_range)[0]
        used_range = True
    elif np.any(finite):
        candidates = np.where(finite)[0]
        used_range = False
    else:
        return int(values.size // 2), False

    local_idx = int(np.argmin(np.abs(values[candidates] - float(target))))
    return int(candidates[local_idx]), used_range


def invert_single_signal_lcurve(
    time_ms: np.ndarray,
    amplitude: np.ndarray,
    *,
    signal_name: str,
    config: Optional[LCurveConfig] = None,
    alpha_values: Optional[np.ndarray] = None,
) -> LCurveInversionResult:
    """Invert one signal and choose regularization using L-curve criterion."""

    cfg = config or LCurveConfig()

    t_arr = np.asarray(time_ms, dtype=float).ravel()
    y_arr = np.asarray(amplitude, dtype=float).ravel()
    valid = np.isfinite(t_arr) & np.isfinite(y_arr) & (t_arr > 0)
    t_valid = t_arr[valid]
    y_valid = y_arr[valid]

    if t_valid.size < cfg.min_points_after_trim:
        raise ValueError(
            f"Signal '{signal_name}' contains {t_valid.size} valid points; "
            f"at least {cfg.min_points_after_trim} are required."
        )

    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    y_valid = y_valid[order]

    if cfg.t2_min_ms <= 0 or cfg.t2_max_ms <= 0 or cfg.t2_max_ms <= cfg.t2_min_ms:
        raise ValueError("Invalid T2 bounds: require 0 < t2_min_ms < t2_max_ms.")

    t2_bins = np.logspace(np.log10(cfg.t2_min_ms), np.log10(cfg.t2_max_ms), int(cfg.num_bins))
    noise = estimate_noise_vector(y_valid)

    if alpha_values is None:
        alpha_values_arr = np.logspace(np.log10(cfg.alpha_min), np.log10(cfg.alpha_max), int(cfg.alpha_count))
    else:
        alpha_values_arr = np.asarray(alpha_values, dtype=float).ravel()

    residual_norms = np.full(alpha_values_arr.shape, np.nan, dtype=float)
    roughness_norms = np.full(alpha_values_arr.shape, np.nan, dtype=float)
    zeta_values = np.full(alpha_values_arr.shape, np.nan, dtype=float)
    eta_values = np.full(alpha_values_arr.shape, np.nan, dtype=float)
    reciprocal_slopes = np.full(alpha_values_arr.shape, np.nan, dtype=float)
    model_matrix = np.full((alpha_values_arr.size, int(cfg.num_bins)), np.nan, dtype=float)
    safe_floor = 1e-30

    for idx, alpha in enumerate(alpha_values_arr):
        model, synthetic, diagnostics = solve_t2_nnls_matlab_compatible(
            data=y_valid,
            time_ms=t_valid,
            noise=noise,
            t2_bins_ms=t2_bins,
            regularization=float(alpha),
        )

        if model.size < int(cfg.num_bins):
            raise RuntimeError(f"Inversion returned invalid model length for alpha index {idx}.")

        spectrum = model[: int(cfg.num_bins)]
        model_matrix[idx, :] = spectrum

        if diagnostics.size >= 4 and np.isfinite(diagnostics[2]) and np.isfinite(diagnostics[3]):
            zeta = float(diagnostics[2])
            eta = float(diagnostics[3])
        else:
            residual_vec = synthetic[:, 2] if synthetic.size > 0 else np.array([], dtype=float)
            zeta = float(np.sum(residual_vec**2))
            eta = float(np.sum(np.diff(spectrum, n=2) ** 2))

        zeta_values[idx] = zeta
        eta_values[idx] = eta
        residual_norms[idx] = float(np.sqrt(max(zeta, 0.0)))
        roughness_norms[idx] = float(np.sqrt(max(eta, 0.0)))

        alpha_paper = float(alpha) ** 2
        reciprocal_slopes[idx] = float((alpha_paper * eta) / max(zeta, safe_floor))

    best_index, used_range = select_best_alpha_by_reciprocal_slope(
        reciprocal_slopes,
        target=float(cfg.slope_reciprocal_target),
        valid_range=cfg.slope_reciprocal_valid_range,
    )

    best_alpha = float(alpha_values_arr[best_index])
    best_spectrum = model_matrix[best_index, :]

    _, best_synthetic, _ = solve_t2_nnls_matlab_compatible(
        data=y_valid,
        time_ms=t_valid,
        noise=noise,
        t2_bins_ms=t2_bins,
        regularization=best_alpha,
    )

    fit_time = best_synthetic[:, 0] if best_synthetic.size > 0 else np.array([], dtype=float)
    fit_amp = best_synthetic[:, 1] if best_synthetic.size > 0 else np.array([], dtype=float)
    residual = best_synthetic[:, 2] if best_synthetic.size > 0 else np.array([], dtype=float)

    return LCurveInversionResult(
        signal_name=signal_name,
        t2_bins_ms=t2_bins,
        spectrum=best_spectrum,
        fit_time_ms=fit_time,
        fit_amplitude=fit_amp,
        residual=residual,
        best_regularization=best_alpha,
        best_index=int(best_index),
        alpha_values=alpha_values_arr,
        residual_norms=residual_norms,
        roughness_norms=roughness_norms,
        zeta_values=zeta_values,
        eta_values=eta_values,
        slope_reciprocal_values=reciprocal_slopes,
        used_range_filter=bool(used_range),
        metadata={
            "slope_reciprocal_target": float(cfg.slope_reciprocal_target),
            "slope_reciprocal_valid_range": tuple(float(v) for v in cfg.slope_reciprocal_valid_range),
        },
    )
