"""Saturation scenarios and water allocation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import numpy as np

from .configs import GeometryConfig


SW_CRIT_INCIRCLE = float(1.0 - np.pi / (3.0 * np.sqrt(3.0)))
SW_RESIDUAL = 0.0226


@dataclass(frozen=True)
class SaturationScenario:
    """Local and global water saturation state for the two-pore benchmark."""

    key: str
    name: str
    sw_large_local: float
    sw_small_local: float
    sw_global: float
    allow_exchange: bool
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_triangle_scenario(
    geometry: GeometryConfig,
    key: str,
    name: str,
    sw_large_local: float,
    sw_small_local: float,
    note: str,
    *,
    allow_exchange: bool = True,
) -> SaturationScenario:
    """Create a clipped local-saturation scenario and compute global Sw."""

    sw_large = float(np.clip(sw_large_local, SW_RESIDUAL, 1.0))
    sw_small = float(np.clip(sw_small_local, SW_RESIDUAL, 1.0))
    water_large = geometry.large_volume_um3 * sw_large
    water_small = geometry.small_volume_um3 * sw_small
    sw_global = (water_large + water_small) / geometry.total_volume_um3

    return SaturationScenario(
        key=key,
        name=name,
        sw_large_local=sw_large,
        sw_small_local=sw_small,
        sw_global=float(sw_global),
        allow_exchange=bool(allow_exchange),
        note=note,
    )


def build_default_triangle_scenarios(geometry: GeometryConfig | None = None) -> List[SaturationScenario]:
    """Return the current validated saturation states from Auto_T2-T23."""

    cfg = geometry or GeometryConfig()
    return [
        make_triangle_scenario(
            cfg,
            "Residual",
            "Residual",
            SW_RESIDUAL,
            SW_RESIDUAL,
            "Both pores keep residual corner water films.",
        ),
        make_triangle_scenario(
            cfg,
            "Sw12p0",
            "Sw 12.0%",
            0.120,
            0.120,
            "Both pores use the same local saturation.",
        ),
        make_triangle_scenario(
            cfg,
            "Sw15p3",
            "Sw 15.3%",
            0.153,
            0.153,
            "Both pores use the same local saturation.",
        ),
        make_triangle_scenario(
            cfg,
            "BothCrit",
            "Both crit",
            SW_CRIT_INCIRCLE,
            SW_CRIT_INCIRCLE,
            "Upper and lower triangles both reach the incircle critical state.",
        ),
        make_triangle_scenario(
            cfg,
            "UpperCrit",
            "Upper crit",
            SW_CRIT_INCIRCLE,
            1.0,
            "Upper large triangle reaches the incircle critical state; lower small triangle is saturated.",
        ),
        make_triangle_scenario(
            cfg,
            "Full",
            "Sw 100.0%",
            1.0,
            1.0,
            "Fully saturated state.",
        ),
    ]


SCENARIO_COLUMNS = [
    "key",
    "name",
    "sw_large_local",
    "sw_small_local",
    "sw_global",
    "allow_exchange",
    "note",
]


def scenarios_to_rows(scenarios: Iterable[SaturationScenario]) -> List[dict]:
    """Convert saturation scenarios to stable row dictionaries."""

    return [{column: scenario.to_dict()[column] for column in SCENARIO_COLUMNS} for scenario in scenarios]
