"""Geometry contracts for pore-scale simulation domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class GeometryMetadata:
    """Summary needed to verify geometry against theory or COMSOL."""

    name: str
    area_um2: float
    depth_um: float
    volume_um3: float
    boundary_markers: Mapping[str, int]


@dataclass(frozen=True)
class TwoPoreGeometryMetadata:
    """Metadata for the current large/small triangle benchmark."""

    large: GeometryMetadata
    small: GeometryMetadata
    throat_length_um: float
    throat_width_um: float


def boundary_marker_contract() -> dict[str, str]:
    """Return the shared meaning of boundary marker categories."""

    return {
        "solid_wall": "Wall boundary where surface relaxivity rho is applied.",
        "water_air_interface": "Internal water/air interface; no surface relaxivity unless explicitly enabled.",
        "throat_or_connection": "Open connection between pore bodies in coupled simulations.",
        "symmetry_or_no_flux": "No-flux boundary used for isolated or symmetry domains.",
    }
