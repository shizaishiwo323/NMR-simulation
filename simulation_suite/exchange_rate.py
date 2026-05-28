"""Equivalent exchange-rate summaries derived from simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExchangeRateEstimate:
    """Equivalent two-site exchange-rate estimate for one saturation state."""

    scenario_key: str
    sw_global: float
    k_large_to_small_per_ms: float
    k_small_to_large_per_ms: float
    method: str
    note: str

    @property
    def k_total_per_ms(self) -> float:
        return self.k_large_to_small_per_ms + self.k_small_to_large_per_ms
