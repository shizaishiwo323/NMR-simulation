"""PDE simulation interfaces for strict pyGIMLi/COMSOL-aligned logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryConditionSummary:
    """Human-readable boundary condition record for one PDE solve."""

    domain_name: str
    initial_condition: str
    diffusion_term: str
    bulk_relaxation: str
    surface_relaxation: str
    coupling_condition: str


@dataclass(frozen=True)
class DecayResult:
    """Signal and metadata returned by a PDE decay solve."""

    time_ms: np.ndarray
    amplitude: np.ndarray
    boundary_summary: BoundaryConditionSummary


def describe_uncoupled_pde() -> BoundaryConditionSummary:
    """Document the intended isolated-pore PDE boundary logic."""

    return BoundaryConditionSummary(
        domain_name="isolated_pore",
        initial_condition="Uniform magnetization in the water-filled pore domain.",
        diffusion_term="D * Laplacian(M) in water phase.",
        bulk_relaxation="-M / T2B throughout the water phase.",
        surface_relaxation="Robin sink rho * M on solid-wall markers.",
        coupling_condition="No exchange flux between large and small pores; signals are volume-weighted after solving.",
    )


def describe_coupled_pde() -> BoundaryConditionSummary:
    """Document the intended coupled-pore PDE boundary logic."""

    return BoundaryConditionSummary(
        domain_name="connected_two_pore_domain",
        initial_condition="Uniform magnetization in all connected water-filled regions.",
        diffusion_term="D * Laplacian(M) over the connected water domain and throat.",
        bulk_relaxation="-M / T2B throughout the water phase.",
        surface_relaxation="Robin sink rho * M on solid-wall markers only.",
        coupling_condition="Exchange emerges from diffusion through the connected throat/interface geometry.",
    )
