"""Top-level package for standardized NMR T2 inversion workflows.

This package provides a clean API for four core capabilities:

1. NNLS inversion
2. L-curve-based regularization selection
3. Plotting and visualization
4. Gaussian peak decomposition

The public API is intentionally small and stable so that notebooks and
scripts can rely on long-term compatibility.
"""

from .config import GaussianConfig, LCurveConfig, NnlsConfig, PlotConfig
from .gaussian import decompose_spectrum_as_gaussians
from .lcurve import invert_single_signal_lcurve
from .models import GaussianDecompositionResult, LCurveInversionResult, NnlsInversionResult, TrimmedSignal
from .nnls import invert_single_signal_nnls
from .pipelines import (
    run_gaussian_decomposition_on_spectrum_workbook,
    run_lcurve_workbook,
    run_nnls_workbook,
    run_plotting_workbook_pair,
)

__all__ = [
    "GaussianConfig",
    "LCurveConfig",
    "NnlsConfig",
    "PlotConfig",
    "TrimmedSignal",
    "NnlsInversionResult",
    "LCurveInversionResult",
    "GaussianDecompositionResult",
    "invert_single_signal_nnls",
    "invert_single_signal_lcurve",
    "decompose_spectrum_as_gaussians",
    "run_nnls_workbook",
    "run_lcurve_workbook",
    "run_plotting_workbook_pair",
    "run_gaussian_decomposition_on_spectrum_workbook",
]
