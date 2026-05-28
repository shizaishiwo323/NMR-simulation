"""Organize current triangle simulation outputs into a reusable case folder.

This script copies existing results only.  It does not delete or move the
original output folders, so previous runs remain traceable.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path("simulation_outputs")

def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if not value:
        return default
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _angle_token(values: list[float]) -> str:
    return "-".join(str(int(v)) if abs(v - int(v)) < 1e-9 else str(v).replace(".", "p") for v in values)


def _theta_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


CONTACT_ANGLE_DEG = _env_float("NMR_CONTACT_ANGLE_DEG", 0.0)
INNER_ANGLES_DEG = _env_float_list("NMR_TRIANGLE_ANGLES_DEG", [60.0, 60.0, 60.0])
CASE_ID = os.environ.get(
    "NMR_CASE_ID",
    f"triangle_CA{int(CONTACT_ANGLE_DEG) if abs(CONTACT_ANGLE_DEG-int(CONTACT_ANGLE_DEG)) < 1e-9 else _theta_token(CONTACT_ANGLE_DEG)}deg_angles{_angle_token(INNER_ANGLES_DEG)}",
)
CASE = ROOT / "cases" / CASE_ID

SUFFIX = f"CA{_theta_token(CONTACT_ANGLE_DEG)}deg_angles_{_angle_token(INNER_ANGLES_DEG)}"
FULL = Path(os.environ.get("NMR_TRI_ROOT", str(ROOT / f"triangle_full_{SUFFIX}")))
FULL_TABLES = FULL / "tables"
FULL_MAPS = FULL / "maps"
FULL_FIGURES = FULL / "figures"
CA_BASE = Path(
    os.environ.get(
        "NMR_CA_BASE",
        str(ROOT / f"triangle_{_angle_token(INNER_ANGLES_DEG).replace('-', '_')}_contact_{int(CONTACT_ANGLE_DEG)}"),
    )
)
COMBINED = Path(os.environ.get("NMR_COMBINED_OUT", str(ROOT / f"combined_figures_{SUFFIX}")))


manifest: list[dict[str, str]] = []


def copy_file(src: Path, dst_dir: Path, module: str, role: str) -> None:
    if not src.exists() or not src.is_file():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    manifest.append(
        {
            "module": module,
            "role": role,
            "source": str(src),
            "organized_path": str(dst),
            "bytes": str(dst.stat().st_size),
        }
    )


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path, module: str, role: str) -> None:
    if not src_dir.exists():
        return
    for src in sorted(src_dir.glob(pattern)):
        copy_file(src, dst_dir, module, role)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    CASE.mkdir(parents=True, exist_ok=True)

    metadata = CASE / "00_metadata"
    water = CASE / "01_water_distribution"
    t2 = CASE / "02_T2"
    t2_tables = t2 / "tables"
    t2_figures = t2 / "figures_per_state"
    t2_validation = t2 / "validation_uncoupled"
    t2_metrics = t2 / "metrics"
    t2t2 = CASE / "03_T2_T2_exchange"
    dt2 = CASE / "04_D_T2"
    combined = CASE / "05_combined_panels"
    effective = CASE / "06_effective_coupling_Zhou2026_GRL"

    # Metadata and run control.
    copy_file(FULL / "Simulation_Control_Used.json", metadata, "metadata", "control_used")
    copy_file(FULL_TABLES / "AllStates_Summary.xlsx", metadata, "metadata", "state_summary")
    copy_file(FULL_TABLES / "Corrected_Table_Audit.xlsx", metadata, "metadata", "correction_audit")
    copy_file(ROOT / "zhou_2026_grl_exchange_text.txt", effective, "effective_coupling", "paper_text_extract")

    case_config = {
        "case_id": CASE_ID,
        "geometry_type": "triangle_two_pore",
        "contact_angle_deg": CONTACT_ANGLE_DEG,
        "inner_angles_deg": INNER_ANGLES_DEG,
        "source_outputs": {
            "full_suite": str(FULL),
            "ca_water_distribution": str(CA_BASE),
            "combined_figures": str(COMBINED),
        },
        "organized_at": datetime.now().isoformat(timespec="seconds"),
        "notes": [
            "Original files are copied, not moved.",
            "T2 corrected tables use component-wise Large/Small spectra and saturation-scaled amplitudes.",
            "Effective coupling follows the Zhou et al. 2026 GRL-style phi_c/lambda_mean parameterization.",
        ],
    }
    write_text(metadata / "case_config.json", json.dumps(case_config, indent=2, ensure_ascii=False))

    # Water distribution / verified CA baseline.
    copy_glob(CA_BASE, "Figure_*.png", water, "water_distribution", "baseline_figures")
    copy_glob(CA_BASE, "*.xlsx", water / "tables", "water_distribution", "baseline_tables")
    copy_file(CA_BASE / "Simulation_Control_Used.json", water / "metadata", "water_distribution", "baseline_control")

    # T2 and decay tables.
    for name in [
        "AllStates_T2_Components_Uncoupled_Coupled_Corrected.xlsx",
        "AllStates_Decay_Components_Uncoupled_Coupled_Corrected.xlsx",
        "AllStates_T2_Uncoupled_Coupled_Corrected.xlsx",
        "AllStates_Decay_Uncoupled_Coupled_Corrected.xlsx",
        "AllStates_T2_Components_Uncoupled_Coupled.xlsx",
        "AllStates_Decay_Components_Uncoupled_Coupled.xlsx",
        "AllStates_T2_Uncoupled_Coupled.xlsx",
        "AllStates_Decay_Uncoupled_Coupled.xlsx",
    ]:
        copy_file(FULL_TABLES / name, t2_tables, "T2", "table")

    copy_glob(FULL_FIGURES, "*_T2_uncoupled_vs_coupled.png", t2_figures, "T2", "per_state_figure")
    copy_file(COMBINED / "Combined_00C_Uncoupled_LargeSmall_Normalized_Validation.png", t2_validation, "T2", "uncoupled_validation")
    copy_file(COMBINED / "Combined_00D_Uncoupled_LargeSmallOnly_Normalized_Validation.png", t2_validation, "T2", "uncoupled_validation")
    copy_file(COMBINED / "Combined_01_AllStates_T2_Uncoupled_vs_Coupled.png", t2 / "combined", "T2", "combined_panel")
    copy_file(COMBINED / "Combined_02_AllStates_Decay_Uncoupled_vs_Coupled.png", t2 / "combined", "T2", "combined_panel")
    copy_file(COMBINED / "Combined_08_Component_T2_LargeSmallTotal.png", t2 / "combined", "T2", "component_panel")
    copy_file(COMBINED / "Combined_09_Component_Decay_LargeSmallTotal.png", t2 / "combined", "T2", "component_panel")
    copy_file(COMBINED / "Combined_03_T2_Metric_Summary.png", t2_metrics, "T2", "metric_figure")
    copy_file(COMBINED / "Combined_T2_Component_Peak_Area_Metrics.xlsx", t2_metrics, "T2", "metric_table")
    copy_file(COMBINED / "T2_Old_Sw12_vs_New_P4_Drainage_Audit.xlsx", t2_metrics, "T2", "validation_audit")

    # T2-T2 exchange maps.
    copy_glob(FULL_MAPS, "*_T2_T2_Map.xlsx", t2t2 / "maps", "T2_T2", "map_table")
    copy_glob(FULL_FIGURES, "*_T2_T2.png", t2t2 / "figures_per_state", "T2_T2", "map_figure")
    copy_file(COMBINED / "Combined_04_AllStates_Coupled_T2T2_Maps.png", t2t2 / "combined", "T2_T2", "combined_panel")

    # D-T2 / PFG-simulated maps.
    copy_glob(FULL_MAPS, "*DT2*.xlsx", dt2 / "maps", "D_T2", "map_or_signal_table")
    copy_glob(FULL_FIGURES, "*DT2*.png", dt2 / "figures_per_state", "D_T2", "map_figure")
    copy_file(COMBINED / "Combined_05_Selected_Strict_DT2_Uncoupled_vs_Coupled.png", dt2 / "combined", "D_T2", "combined_panel")

    # Effective exchange model based on GRL 2026.
    copy_file(COMBINED / "Combined_00E_EffectiveCoupled_LargeSmallOnly_Zhou2026GRLScaled.png", effective / "figures", "effective_coupling", "combined_panel")
    copy_file(COMBINED / "EffectiveCoupled_T2_Zhou2026GRLScaled.xlsx", effective / "tables", "effective_coupling", "t2_table")
    copy_file(COMBINED / "EffectiveCoupled_Zhou2026GRLScaled_Parameters.xlsx", effective / "tables", "effective_coupling", "parameter_table")

    # Current combined panels kept as an overview collection.
    copy_glob(COMBINED, "Combined_*.png", combined, "combined", "figure")
    copy_glob(COMBINED, "*.xlsx", combined / "tables", "combined", "table")

    readme = f"""# {CASE_ID}

This folder is the organized case output for:

- geometry: two equilateral triangular pores
- contact angle: CA = {CONTACT_ANGLE_DEG:g} deg
- triangle inner angles: {' / '.join(f'{v:g}' for v in INNER_ANGLES_DEG)} deg

Folder layout:

- `00_metadata`: run control, state summary, correction audit, case config.
- `01_water_distribution`: water retention and morphology baseline from the CA simulation.
- `02_T2`: T2 decay/spectra tables, per-state figures, validation plots, and component metrics.
- `03_T2_T2_exchange`: T2-store-T2 exchange maps and combined panels.
- `04_D_T2`: PFG-simulated D-T2 maps/signals and combined panels.
- `05_combined_panels`: overview figures/tables copied from the current combined figure folder.
- `06_effective_coupling_Zhou2026_GRL`: effective exchange-coupled T2 results using the GRL 2026 style phi_c/lambda_mean parameterization.

Original output folders are not modified.  Use `manifest.csv` to trace every organized file back to its source.
"""
    write_text(CASE / "README.md", readme)

    manifest_path = CASE / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["module", "role", "source", "organized_path", "bytes"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Organized case folder: {CASE}")
    print(f"Copied files: {len(manifest)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
