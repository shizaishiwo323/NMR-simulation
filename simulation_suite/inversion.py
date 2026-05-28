"""Unified inversion wrappers used by simulation runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import LCurveConfig, NnlsConfig
from ..lcurve import invert_single_signal_lcurve
from ..nnls import invert_single_signal_nnls
from .configs import InversionMode, InversionSettings


@dataclass(frozen=True)
class T2InversionOutput:
    """Small stable result object for one T2 inversion."""

    t2_ms: np.ndarray
    amplitude: np.ndarray
    selected_alpha: float
    mode: str


def invert_t2_signal(
    time_ms: np.ndarray,
    signal: np.ndarray,
    settings: InversionSettings,
    *,
    signal_name: str = "signal",
) -> T2InversionOutput:
    """Invert a single decay signal using fixed-alpha NNLS or L-curve."""

    mode = InversionMode(settings.mode)
    if mode is InversionMode.L_CURVE:
        result = invert_single_signal_lcurve(
            np.asarray(time_ms, dtype=float),
            np.asarray(signal, dtype=float),
            signal_name=signal_name,
            config=LCurveConfig(
                num_bins=int(settings.t2_bins),
                t2_min_ms=float(settings.t2_min_ms),
                t2_max_ms=float(settings.t2_max_ms),
            ),
        )
        return T2InversionOutput(
            t2_ms=result.t2_bins_ms,
            amplitude=result.spectrum,
            selected_alpha=float(result.best_alpha),
            mode=mode.value,
        )

    result = invert_single_signal_nnls(
        np.asarray(time_ms, dtype=float),
        np.asarray(signal, dtype=float),
        signal_name=signal_name,
        config=NnlsConfig(
            num_bins=int(settings.t2_bins),
            regularization=float(settings.fixed_alpha),
            t2_min_ms=float(settings.t2_min_ms),
            t2_max_ms=float(settings.t2_max_ms),
        ),
    )
    return T2InversionOutput(
        t2_ms=result.t2_bins_ms,
        amplitude=result.spectrum,
        selected_alpha=float(settings.fixed_alpha),
        mode=mode.value,
    )
