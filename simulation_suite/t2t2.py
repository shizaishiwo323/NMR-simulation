"""T2-store-T2 exchange-map interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class T2T2Result:
    """Container for a T2-T2 signal and inverted exchange map."""

    t1_axis_ms: np.ndarray
    t2_axis_ms: np.ndarray
    signal: np.ndarray
    map_amplitude: np.ndarray
    mixing_time_ms: float
    alpha: float


def off_diagonal_ratio(map_amplitude: np.ndarray, t2_axis_ms: np.ndarray, diag_width_decades: float = 0.15) -> float:
    """Estimate the fraction of T2-T2 intensity away from the diagonal."""

    values = np.asarray(map_amplitude, dtype=float)
    axis = np.asarray(t2_axis_ms, dtype=float)
    if values.ndim != 2 or values.shape[0] != axis.size or values.shape[1] != axis.size:
        raise ValueError("map_amplitude must be a square matrix matching t2_axis_ms.")
    total = float(np.sum(np.maximum(values, 0.0)))
    if total <= 0.0:
        return 0.0

    log_axis = np.log10(axis)
    diagonal_mask = np.abs(log_axis[:, None] - log_axis[None, :]) <= float(diag_width_decades)
    diagonal = float(np.sum(np.maximum(values, 0.0)[diagonal_mask]))
    return max(0.0, 1.0 - diagonal / total)
