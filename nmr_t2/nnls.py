"""NNLS inversion algorithms for T2 spectrum reconstruction.

This module keeps numerical behavior close to the legacy MATLAB workflow
while exposing Pythonic, typed interfaces.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter

from .config import NnlsConfig
from .models import NnlsInversionResult


def build_exponential_kernel(time_ms: np.ndarray, t2_bins_ms: np.ndarray) -> np.ndarray:
    """Build the exponential decay kernel $K_{ij}=\exp(-t_i/T2_j)$."""

    t_col = np.asarray(time_ms, dtype=float).reshape(-1, 1)
    t2_row = np.asarray(t2_bins_ms, dtype=float).reshape(1, -1)
    return np.exp(-t_col / t2_row)


def _std_matlab_style(values: np.ndarray) -> float:
    """Match MATLAB `std(x)` default behavior (normalization by $N-1$)."""

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _movmean_matlab_style(values: np.ndarray, window: int) -> np.ndarray:
    """Match MATLAB `movmean(x, k)` endpoint behavior (shrink windows)."""

    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n == 0:
        return np.array([], dtype=float)

    k = int(window)
    if k <= 1:
        return arr.copy()

    back = (k - 1) // 2
    forward = k - back - 1

    idx = np.arange(n)
    left = np.maximum(0, idx - back)
    right = np.minimum(n - 1, idx + forward)

    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    window_sum = csum[right + 1] - csum[left]
    window_count = (right - left + 1).astype(float)
    return window_sum / window_count


def estimate_noise_vector(signal: np.ndarray) -> np.ndarray:
    """Estimate a noise vector for weighted inversion.

    The estimation follows the same strategy used in the legacy MATLAB wrapper:
    remove a moving-average trend and compute residual noise.
    """

    signal_arr = np.asarray(signal, dtype=float).ravel()
    n = signal_arr.size
    if n == 0:
        return np.array([], dtype=float)

    window = max(5, min(21, n // 8))
    trend = _movmean_matlab_style(signal_arr, window)
    noise = signal_arr - trend

    sigma = _std_matlab_style(noise)
    if (not np.isfinite(sigma)) or sigma <= 0:
        noise = np.zeros_like(signal_arr)
        noise[-1] = 1.0
    return noise


def _build_regularization_matrix_matlab_compatible(model_size: int) -> np.ndarray:
    """Build the same regularization matrix as the MATLAB implementation."""

    reg = np.diag(-2.0 * np.ones(model_size))
    reg += np.diag(np.ones(model_size - 1), 1)
    reg += np.diag(np.ones(model_size - 1), -1)

    middle = np.hstack([reg[1 : model_size - 2, 0 : model_size - 1], np.zeros((model_size - 3, 1))])
    top1 = np.hstack([np.array([1.0]), np.zeros(model_size - 1)])
    top2 = np.hstack([np.array([-2.0, 1.0]), np.zeros(model_size - 2)])
    bottom = np.hstack([np.zeros(model_size - 3), np.array([1.0, -2.0, 1.0])])

    return np.vstack([top1, top2, middle, bottom])


def solve_t2_nnls_matlab_compatible(
    data: np.ndarray,
    time_ms: np.ndarray,
    noise: np.ndarray,
    t2_bins_ms: np.ndarray,
    regularization: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve T2 NNLS inversion with MATLAB-compatible preprocessing.

    Returns
    -------
    model:
        NNLS model vector where the last element is a baseline term.
    synthetic:
        A `(n_resampled, 3)` table: time, fitted data, residual.
    diagnostics:
        A length-4 vector:
        `[weighted_residual_norm, data_tolerance, zeta, eta]`.
    """

    data_arr = np.asarray(data, dtype=float).ravel()
    time_arr = np.asarray(time_ms, dtype=float).ravel()
    noise_arr = np.asarray(noise, dtype=float).ravel()
    t2_arr = np.asarray(t2_bins_ms, dtype=float).ravel()

    if not (data_arr.size == time_arr.size == noise_arr.size):
        raise ValueError("`data`, `time_ms`, and `noise` must have identical lengths.")
    if data_arr.size == 0 or t2_arr.size == 0:
        return np.array([]), np.empty((0, 3), dtype=float), np.array([np.nan, np.nan], dtype=float)

    sigma = _std_matlab_style(noise_arr)
    downsample_factor = 20

    # Step 1: clip heavily decayed tail using moving-average threshold.
    clip_window = 100
    filter_kernel = np.ones(clip_window, dtype=float) / float(clip_window)
    conv_full = np.convolve(filter_kernel, data_arr, mode="full")
    below_threshold = np.where(conv_full < (data_arr[0] / 1e5))[0] + 1
    end_index = int(np.min(below_threshold)) if below_threshold.size > 0 else data_arr.size
    end_index = min(end_index, data_arr.size)

    data_arr = data_arr[:end_index]
    time_arr = time_arr[:end_index]

    # Step 2: smooth with reverse filtering to mirror MATLAB behavior.
    smooth_window = 5
    reversed_data = data_arr[::-1]
    reversed_smoothed = lfilter(np.ones(smooth_window, dtype=float) / float(smooth_window), [1.0], reversed_data)

    if reversed_smoothed.size >= smooth_window:
        data_arr = reversed_smoothed[::-1][smooth_window - 1 :]
        keep_count = time_arr.size - smooth_window + 1
        time_arr = time_arr[:keep_count]
    else:
        data_arr = np.array([], dtype=float)
        time_arr = np.array([], dtype=float)

    if data_arr.size == 0 or time_arr.size == 0:
        model = np.zeros(t2_arr.size + 1, dtype=float)
        return model, np.empty((0, 3), dtype=float), np.array([np.nan, np.nan], dtype=float)

    # Step 3: logarithmic downsampling.
    n_log = max(1, int(np.floor(time_arr.size / downsample_factor)))
    target_time = np.logspace(np.log10(time_arr[-1]), np.log10(time_arr[0]), n_log)[::-1]

    sampled_time = np.zeros(n_log, dtype=float)
    sampled_data = np.zeros(n_log, dtype=float)
    sampled_sigma = np.zeros(n_log, dtype=float)

    sampled_time[0] = time_arr[0]
    sampled_data[0] = data_arr[0]
    sampled_sigma[0] = sigma

    last_index = 1  # MATLAB-style one-based index bookkeeping.
    for idx in range(1, n_log):
        nearest_indices = np.where(np.abs(time_arr - target_time[idx]) == np.min(np.abs(time_arr - target_time[idx])))[0] + 1
        if nearest_indices.size == 0:
            sampled_time[idx] = np.nan
            sampled_data[idx] = np.nan
            sampled_sigma[idx] = np.nan
            continue

        current_index = int(nearest_indices[0])
        if current_index == last_index:
            sampled_time[idx] = np.nan
            sampled_data[idx] = np.nan
            sampled_sigma[idx] = np.nan
        else:
            segment = slice(last_index, current_index)
            sampled_time[idx] = float(np.mean(time_arr[segment]))
            sampled_data[idx] = float(np.mean(data_arr[segment]))
            sampled_sigma[idx] = float(sigma / np.sqrt(current_index - last_index))

        last_index = current_index

    valid = np.where(np.abs(sampled_time) > 0)[0]
    reduced_time = sampled_time[valid]
    reduced_data = sampled_data[valid]
    reduced_sigma = sampled_sigma[valid]

    if reduced_time.size == 0:
        model = np.zeros(t2_arr.size + 1, dtype=float)
        return model, np.empty((0, 3), dtype=float), np.array([np.nan, np.nan], dtype=float)

    rhs_data = reduced_data / reduced_sigma

    # Step 4: build weighted forward operator with baseline term.
    n_t2 = t2_arr.size
    model_size = n_t2 + 1
    decay = np.exp(-reduced_time[:, None] / t2_arr[None, :])

    forward_unweighted = np.zeros((reduced_time.size, model_size), dtype=float)
    forward_weighted = np.zeros((reduced_time.size, model_size), dtype=float)
    forward_unweighted[:, :n_t2] = decay
    forward_weighted[:, :n_t2] = decay / reduced_sigma[:, None]
    forward_unweighted[:, n_t2] = 1.0
    forward_weighted[:, n_t2] = 1.0 / reduced_sigma

    reg_matrix = _build_regularization_matrix_matlab_compatible(model_size)
    system_matrix = np.vstack([forward_weighted, float(regularization) * reg_matrix])
    rhs = np.concatenate([rhs_data, np.zeros(model_size, dtype=float)])

    model, _ = nnls(system_matrix, rhs)

    fitted = forward_unweighted @ model
    residual = reduced_data - fitted
    synthetic = np.column_stack([reduced_time, fitted, residual])

    sigma_safe = np.where((~np.isfinite(reduced_sigma)) | (np.abs(reduced_sigma) <= 1e-30), 1.0, reduced_sigma)
    weighted_residual = residual / sigma_safe
    zeta = float(np.sum(weighted_residual**2))
    eta = float(np.sum((reg_matrix @ model) ** 2))
    diagnostics = np.array([np.linalg.norm(weighted_residual), np.sqrt(np.sum(sigma_safe**2)), zeta, eta], dtype=float)

    return model, synthetic, diagnostics


def invert_single_signal_nnls(
    time_ms: np.ndarray,
    amplitude: np.ndarray,
    *,
    signal_name: str,
    config: Optional[NnlsConfig] = None,
    noise: Optional[np.ndarray] = None,
) -> NnlsInversionResult:
    """Invert one signal using fixed-regularization NNLS.

    This function assumes the input signal has already been cleaned/sorted and,
    if desired, trimmed from the global peak.
    """

    cfg = config or NnlsConfig()

    time_arr = np.asarray(time_ms, dtype=float).ravel()
    signal_arr = np.asarray(amplitude, dtype=float).ravel()
    valid = np.isfinite(time_arr) & np.isfinite(signal_arr) & (time_arr > 0)

    t_valid = time_arr[valid]
    y_valid = signal_arr[valid]
    if t_valid.size < 2:
        raise ValueError(f"Signal '{signal_name}' has fewer than two valid points.")

    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    y_valid = y_valid[order]

    if cfg.t2_min_ms <= 0 or cfg.t2_max_ms <= 0 or cfg.t2_max_ms <= cfg.t2_min_ms:
        raise ValueError("Invalid T2 bounds: require 0 < t2_min_ms < t2_max_ms.")

    t2_bins = np.logspace(np.log10(cfg.t2_min_ms), np.log10(cfg.t2_max_ms), int(cfg.num_bins))

    if noise is None:
        noise_vec = estimate_noise_vector(y_valid)
    else:
        noise_arr = np.asarray(noise, dtype=float).ravel()
        if noise_arr.size != time_arr.size:
            raise ValueError("`noise` must have the same length as the provided `time_ms`.")
        noise_vec = noise_arr[valid][order]

    model, synthetic, diagnostics = solve_t2_nnls_matlab_compatible(
        data=y_valid,
        time_ms=t_valid,
        noise=noise_vec,
        t2_bins_ms=t2_bins,
        regularization=float(cfg.regularization),
    )

    if model.size == 0:
        spectrum = np.zeros(int(cfg.num_bins), dtype=float)
        fit_time = np.array([], dtype=float)
        fit_signal = np.array([], dtype=float)
        residual = np.array([], dtype=float)
        residual_norm = np.nan
        roughness_norm = np.nan
    else:
        spectrum = model[: int(cfg.num_bins)]
        fit_time = synthetic[:, 0] if synthetic.size > 0 else np.array([], dtype=float)
        fit_signal = synthetic[:, 1] if synthetic.size > 0 else np.array([], dtype=float)
        residual = synthetic[:, 2] if synthetic.size > 0 else np.array([], dtype=float)
        residual_norm = float(np.sqrt(max(diagnostics[2], 0.0))) if diagnostics.size >= 4 else np.nan
        roughness_norm = float(np.sqrt(max(diagnostics[3], 0.0))) if diagnostics.size >= 4 else np.nan

    return NnlsInversionResult(
        signal_name=signal_name,
        t2_bins_ms=t2_bins,
        spectrum=spectrum,
        fit_time_ms=fit_time,
        fit_amplitude=fit_signal,
        residual=residual,
        regularization=float(cfg.regularization),
        residual_norm=residual_norm,
        roughness_norm=roughness_norm,
        metadata={"num_bins": int(cfg.num_bins), "t2_min_ms": float(cfg.t2_min_ms), "t2_max_ms": float(cfg.t2_max_ms)},
    )
