"""Reusable CLI entry point for pnextract pore-size vs NMR T2 overlays.

The implementation lives in ``sample16_pnextract_t2_overlay`` because that
script records the validated Sample 16 defaults. This module provides a
sample-agnostic command name for future runs and documentation.
"""

from __future__ import annotations

from advanced_tools.sample16_pnextract_t2_overlay import main


if __name__ == "__main__":
    main()
