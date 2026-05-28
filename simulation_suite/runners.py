"""Command-line runners for the unified NMR simulation suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from .configs import InversionMode, InversionSettings, OutputSettings, SimulationSuiteConfig
from .exports import try_write_excel_workbook, write_csv_rows, write_json
from .saturation import (
    SCENARIO_COLUMNS,
    SW_CRIT_INCIRCLE,
    SW_RESIDUAL,
    build_default_triangle_scenarios,
    scenarios_to_rows,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = Path(r"C:\Users\Yu\Downloads\NMR_T2_DT2_RIGOROUS_T2T2_EXCHANGE_RATE.py")
REFERENCE_PAPER = Path(r"F:\4. Work\1. Diff CA in Imbibiton and Drainage\hess-19-2763-2015.pdf")

HISTORICAL_WORKBOOKS = [
    "Sw_Coupling_Parameters.xlsx",
    "Critical_Sw_Coupling_Parameters.xlsx",
    "T2_Inversion_Detailed_Sw_Coupled.xlsx",
    "T2_Inversion_Detailed_Critical_Sw_Coupled.xlsx",
    "T2_T2_Exchange_Maps_Sw_Coupled.xlsx",
    "T2_T2_Exchange_Maps_Critical_Sw_Coupled.xlsx",
    "Triangle_Raw_Decay_Sw_Coupled.xlsx",
    "Triangle_Raw_Decay_Critical_Sw_Coupled.xlsx",
]

INVENTORY_COLUMNS = ["kind", "path", "exists", "bytes"]
CONFIG_COLUMNS = ["field", "value"]


def build_reference_inventory() -> list[dict]:
    """Inventory local reference files and historical result workbooks."""

    rows = [
        {
            "kind": "reference_script",
            "path": str(REFERENCE_SCRIPT),
            "exists": REFERENCE_SCRIPT.exists(),
            "bytes": REFERENCE_SCRIPT.stat().st_size if REFERENCE_SCRIPT.exists() else None,
        },
        {
            "kind": "reference_paper",
            "path": str(REFERENCE_PAPER),
            "exists": REFERENCE_PAPER.exists(),
            "bytes": REFERENCE_PAPER.stat().st_size if REFERENCE_PAPER.exists() else None,
        },
    ]
    for name in HISTORICAL_WORKBOOKS:
        path = PROJECT_DIR / name
        rows.append(
            {
                "kind": "historical_workbook",
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return rows


def flatten_config_for_table(payload: dict) -> list[dict]:
    """Flatten nested metadata for simple CSV/Excel output."""

    rows: list[dict] = []

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            rows.append({"field": prefix, "value": value})

    visit("", payload)
    return rows


def create_bootstrap_plan(config: SimulationSuiteConfig) -> Dict[str, Path]:
    """Create the first standardized run folder without executing PDE solves."""

    config.output.ensure_dirs()
    scenarios = build_default_triangle_scenarios(config.geometry)
    scenario_rows = scenarios_to_rows(scenarios)
    inventory_rows = build_reference_inventory()

    metadata = config.to_serializable_dict()
    metadata["project_dir"] = str(PROJECT_DIR)
    metadata["sw_residual"] = SW_RESIDUAL
    metadata["sw_crit_incircle"] = SW_CRIT_INCIRCLE
    metadata["purpose"] = (
        "Bootstrap run folder for later T2, T2-T2, D-T2, PDE, and exchange-rate simulations."
    )
    config_rows = flatten_config_for_table(metadata)

    paths = {
        "config_json": write_json(config.output.metadata_dir / "config.json", metadata),
        "scenario_csv": write_csv_rows(config.output.table_dir / "saturation_scenarios.csv", scenario_rows, SCENARIO_COLUMNS),
        "inventory_csv": write_csv_rows(config.output.table_dir / "reference_inventory.csv", inventory_rows, INVENTORY_COLUMNS),
        "config_csv": write_csv_rows(config.output.table_dir / "config_summary.csv", config_rows, CONFIG_COLUMNS),
    }
    workbook_path = try_write_excel_workbook(
        config.output.run_dir / "simulation_bootstrap_plan.xlsx",
        {
            "SaturationScenarios": (scenario_rows, SCENARIO_COLUMNS),
            "ReferenceInventory": (inventory_rows, INVENTORY_COLUMNS),
            "ConfigSummary": (config_rows, CONFIG_COLUMNS),
        },
    )
    if workbook_path is not None:
        paths["bootstrap_workbook"] = workbook_path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified NMR simulation suite runner.")
    parser.add_argument("--task", choices=["plan"], default="plan")
    parser.add_argument("--output-root", default="simulation_outputs")
    parser.add_argument("--run-name", default="bootstrap")
    parser.add_argument("--inversion-mode", choices=[mode.value for mode in InversionMode], default=InversionMode.FIXED.value)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationSuiteConfig(
        inversion=InversionSettings(
            mode=InversionMode(args.inversion_mode),
            fixed_alpha=float(args.alpha),
        ),
        output=OutputSettings(
            root_dir=Path(args.output_root),
            run_name=str(args.run_name),
        ),
    )

    if args.task == "plan":
        paths = create_bootstrap_plan(config)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    print("Simulation suite task completed.")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
