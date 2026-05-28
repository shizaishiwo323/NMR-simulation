"""Batch runner for full triangle contact-angle simulations.

Each case runs the strict full suite first, then generates the combined
figures, validation panels, effective-coupling panel, and organized case folder.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from simulation_control import SETTINGS
import triangle_contact_angle_full_suite as full_suite


ROOT = Path("simulation_outputs")


def angle_token(values: list[float]) -> str:
    return "-".join(str(int(v)) if abs(v - int(v)) < 1e-9 else str(v).replace(".", "p") for v in values)


def theta_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def case_id(contact_angle: float, triangle_angles: list[float]) -> str:
    theta = str(int(contact_angle)) if abs(contact_angle - int(contact_angle)) < 1e-9 else theta_token(contact_angle)
    return f"triangle_CA{theta}deg_angles{angle_token(triangle_angles)}"


def full_output_dir(contact_angle: float, triangle_angles: list[float]) -> Path:
    return ROOT / f"triangle_full_CA{theta_token(contact_angle)}deg_angles_{angle_token(triangle_angles)}"


def combined_output_dir(contact_angle: float, triangle_angles: list[float]) -> Path:
    return ROOT / f"combined_figures_CA{theta_token(contact_angle)}deg_angles_{angle_token(triangle_angles)}"


def configure_case(contact_angle: float, triangle_angles: list[float]) -> None:
    SETTINGS["triangle_ca"]["contact_angle_deg"] = float(contact_angle)
    SETTINGS["triangle_ca"]["triangle_angles_deg"] = [float(v) for v in triangle_angles]
    SETTINGS["geometry"]["contact_angle_deg"] = float(contact_angle)
    SETTINGS["geometry"]["triangle_angles_deg"] = [float(v) for v in triangle_angles]
    SETTINGS["triangle_full"]["output_dir"] = str(full_output_dir(contact_angle, triangle_angles))


def run_subprocess(script: str, contact_angle: float, triangle_angles: list[float]) -> None:
    env = os.environ.copy()
    env["NMR_CONTACT_ANGLE_DEG"] = str(float(contact_angle))
    env["NMR_TRIANGLE_ANGLES_DEG"] = ",".join(str(float(v)) for v in triangle_angles)
    env["NMR_TRI_ROOT"] = str(full_output_dir(contact_angle, triangle_angles))
    env["NMR_COMBINED_OUT"] = str(combined_output_dir(contact_angle, triangle_angles))
    env["NMR_CASE_ID"] = case_id(contact_angle, triangle_angles)
    subprocess.run([sys.executable, script], check=True, env=env)


def run_case(contact_angle: float, triangle_angles: list[float], skip_postprocess: bool, skip_organize: bool) -> None:
    start = time.time()
    print("=" * 72, flush=True)
    print(f"Starting triangle case: CA={contact_angle:g} deg, angles={triangle_angles}", flush=True)
    configure_case(contact_angle, triangle_angles)
    full_suite.main()
    print(f"Full PDE suite finished in {(time.time() - start) / 60.0:.1f} min", flush=True)

    if not skip_postprocess:
        for script in [
            "make_combined_figures.py",
            "make_uncoupled_validation_figure.py",
            "make_effective_coupled_validation_figure.py",
        ]:
            print(f"Postprocessing with {script}", flush=True)
            run_subprocess(script, contact_angle, triangle_angles)

    if not skip_organize:
        print("Organizing output case folder", flush=True)
        run_subprocess("organize_triangle_outputs.py", contact_angle, triangle_angles)

    print(f"Case finished: {case_id(contact_angle, triangle_angles)} in {(time.time() - start) / 60.0:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full triangle contact-angle cases.")
    parser.add_argument("--contact-angles", nargs="+", type=float, default=[30.0, 45.0])
    parser.add_argument("--triangle-angles", nargs=3, type=float, default=[60.0, 60.0, 60.0])
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-organize", action="store_true")
    args = parser.parse_args()

    for contact_angle in args.contact_angles:
        run_case(contact_angle, args.triangle_angles, args.skip_postprocess, args.skip_organize)


if __name__ == "__main__":
    main()
