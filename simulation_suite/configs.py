"""Configuration objects for unified NMR simulation workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


class InversionMode(str, Enum):
    """Supported T2 inversion regularization modes."""

    FIXED = "fixed"
    L_CURVE = "l_curve"


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical parameters used by the current pyGIMLi simulations.

    Units follow the existing scripts: length in um and time in ms.
    """

    diffusion_um2_per_ms: float = 2.0
    bulk_t2_ms: float = 3000.0
    surface_relaxivity_um_per_ms: float = 0.005
    dt_ms: float = 2.0
    t_max_ms: float = 2000.0

    def time_axis_ms(self) -> np.ndarray:
        """Return the PDE/inversion time axis in milliseconds."""

        return np.arange(0.0, self.t_max_ms + self.dt_ms, self.dt_ms)


@dataclass(frozen=True)
class GeometryConfig:
    """Geometry parameters for the current two-triangle benchmark model."""

    large_side_um: float = 20.0
    small_side_um: float = 8.0
    large_depth_um: float = 20.0
    small_depth_um: float = 8.0
    throat_length_um: float = 10.0
    throat_width_um: float = 1.0
    critical_radius_factor: float = 0.985

    @property
    def large_area_um2(self) -> float:
        return float(np.sqrt(3.0) / 4.0 * self.large_side_um**2)

    @property
    def small_area_um2(self) -> float:
        return float(np.sqrt(3.0) / 4.0 * self.small_side_um**2)

    @property
    def large_volume_um3(self) -> float:
        return self.large_area_um2 * self.large_depth_um

    @property
    def small_volume_um3(self) -> float:
        return self.small_area_um2 * self.small_depth_um

    @property
    def total_volume_um3(self) -> float:
        return self.large_volume_um3 + self.small_volume_um3

    @property
    def volume_weights(self) -> Tuple[float, float]:
        total = self.total_volume_um3
        return self.large_volume_um3 / total, self.small_volume_um3 / total


@dataclass(frozen=True)
class InversionSettings:
    """Regularization settings shared by T2, T2-T2, and D-T2 workflows."""

    mode: InversionMode = InversionMode.FIXED
    fixed_alpha: float = 1.0
    t2_min_ms: float = 1.0
    t2_max_ms: float = 1.0e4
    t2_bins: int = 150
    t2t2_alpha: float = 0.05
    dt2_alpha: float = 0.03

    def t2_axis_ms(self) -> np.ndarray:
        """Return the logarithmic T2 axis in milliseconds."""

        return np.logspace(np.log10(self.t2_min_ms), np.log10(self.t2_max_ms), int(self.t2_bins))


@dataclass(frozen=True)
class OutputSettings:
    """Output locations for simulation artifacts."""

    root_dir: Path = Path("simulation_outputs")
    run_name: str = "bootstrap"

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_name

    @property
    def figure_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def metadata_dir(self) -> Path:
        return self.run_dir / "metadata"

    def ensure_dirs(self) -> None:
        for directory in (self.run_dir, self.figure_dir, self.table_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SimulationSuiteConfig:
    """Top-level configuration for a reproducible simulation run."""

    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    inversion: InversionSettings = field(default_factory=InversionSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def to_serializable_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["inversion"]["mode"] = self.inversion.mode.value
        data["output"]["root_dir"] = str(self.output.root_dir)
        return data
