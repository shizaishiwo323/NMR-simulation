"""D-T2/PFG simulation interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DT2Result:
    """Container for a D-T2 signal and inverted map."""

    b_axis: np.ndarray
    te_axis_ms: np.ndarray
    d_axis_um2_per_ms: np.ndarray
    t2_axis_ms: np.ndarray
    signal: np.ndarray
    map_amplitude: np.ndarray
    alpha: float
