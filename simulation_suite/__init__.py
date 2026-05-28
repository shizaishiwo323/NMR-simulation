"""Reusable NMR simulation helpers.

This package intentionally contains only the NMR-related simulation-suite
modules. CT saturation-generation modules are not included in this release.
"""

from .configs import (
    GeometryConfig,
    InversionMode,
    InversionSettings,
    OutputSettings,
    PhysicalParameters,
    SimulationSuiteConfig,
)
from .saturation import SaturationScenario, build_default_triangle_scenarios

__all__ = [
    "GeometryConfig",
    "InversionMode",
    "InversionSettings",
    "OutputSettings",
    "PhysicalParameters",
    "SimulationSuiteConfig",
    "SaturationScenario",
    "build_default_triangle_scenarios",
]
