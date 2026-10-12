"""Validate the shared-radius NMR--SIP joint inversion on straight cylinders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import j0, j1


RADII_UM = np.array([2.0, 5.0, 10.0, 20.0])
POROSITY = 0.5
LENGTH_FACTOR = 5.0

NMR_DIFFUSION_UM2_MS = 2.0
SURFACE_RELAXIVITY_UM_MS = 0.005
NMR_BULK_T2_MS = math.inf
NMR_DT_MS = 2.0
NMR_MAX_TIME_MS = 6000.0

ION_DIFFUSION_M2_S = 1.3e-9
SURFACE_CONDUCTANCE_S = 1.3e-9
WATER_CONDUCTIVITY_S_M = 0.043
EPSILON_0_F_M = 8.85e-12
WATER_RELATIVE_PERMITTIVITY = 80.0
SOLID_RELATIVE_PERMITTIVITY = 7.0
FREQUENCIES_HZ = np.logspace(-3.0, 5.0, 50)

NMR_SHAPE_FACTOR = 2.0  # Infinite circular cylinder: S/V = 2/R.
RADIUS_GRID_UM = np.geomspace(0.5, 50.0, 81)
BETA_GRID = np.logspace(-3.0, 3.0, 7)
LAMBDA_GRID = np.logspace(-6.0, 4.0, 6)
INTERMEDIATE_LAMBDA = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def bessel_t2_ms(radius_um: float, bulk_t2_ms: float = NMR_BULK_T2_MS) -> tuple[float, float]:
    """Return the circular-cylinder surface and total dominant T2 modes."""

    kappa = SURFACE_RELAXIVITY_UM_MS * radius_um / NMR_DIFFUSION_UM2_MS
    xi = brentq(lambda value: value * j1(value) / j0(value) - kappa, 1e-12, 2.404825557 - 1e-9)
    surface_t2_ms = radius_um**2 / (NMR_DIFFUSION_UM2_MS * xi**2)
    total_t2_ms = surface_t2_ms
    if math.isfinite(bulk_t2_ms):
        total_t2_ms = 1.0 / (1.0 / surface_t2_ms + 1.0 / bulk_t2_ms)
    return surface_t2_ms, total_t2_ms


def fit_single_t2_ms(times_ms: np.ndarray, normalized_signal: np.ndarray) -> float:
    mask = (normalized_signal < 0.9) & (normalized_signal > 0.05)
    if np.count_nonzero(mask) < 10:
        raise RuntimeError("NMR decay does not contain enough points between 0.05 and 0.9")
    slope = np.polyfit(times_ms[mask], np.log(normalized_signal[mask]), 1)[0]
    if slope >= 0:
        raise RuntimeError("NMR decay fit did not produce a negative slope")
    return float(-1.0 / slope)


def simulate_nmr(radius_um: float, mesh_path: Path) -> tuple[pd.DataFrame, dict[str, float | int]]:
    import pygimli.meshtools as mt

    from advanced_tools.triangle_contact_angle_full_suite import solve_single_decay

    target_area = math.pi * radius_um**2 / 900.0
    geometry = mt.createCircle(
        pos=[0.0, 0.0],
        radius=radius_um,
        nSegments=96,
        area=target_area,
        marker=1,
        boundaryMarker=1,
    )
    mesh = mt.createMesh(geometry, area=target_area, quality=33)
    mesh.save(str(mesh_path))
    times_ms = np.arange(0.0, NMR_MAX_TIME_MS, NMR_DT_MS)
    signal = solve_single_decay(
        mesh,
        D=NMR_DIFFUSION_UM2_MS,
        T_bulk=NMR_BULK_T2_MS,
        rho=SURFACE_RELAXIVITY_UM_MS,
        times_axis=times_ms,
        step_dt=NMR_DT_MS,
    )
    normalized = signal / signal[0]
    area_um2 = float(np.sum(np.asarray(mesh.cellSizes())))
    perimeter_um = float(sum(boundary.size() for boundary in mesh.boundaries() if boundary.marker() == 1))
    lambda_um = 2.0 * area_um2 / perimeter_um
    surface_t2_ms, exact_t2_ms = bessel_t2_ms(radius_um)
    fit_t2_ms = fit_single_t2_ms(times_ms, normalized)
    frame = pd.DataFrame(
        {
            "radius_um": radius_um,
            "time_ms": times_ms,
            "signal_normalized": normalized,
        }
    )
    summary = {
        "mesh_nodes": int(mesh.nodeCount()),
        "mesh_cells": int(mesh.cellCount()),
        "mesh_area_um2": area_um2,
        "mesh_perimeter_um": perimeter_um,
        "lambda_mesh_um": lambda_um,
        "t2_surface_bessel_ms": surface_t2_ms,
        "t2_exact_ms": exact_t2_ms,
        "t2_fit_ms": fit_t2_ms,
    }
    return frame, summary


def run_ac3d_unit_field(sip_root: Path, mask_path: Path, grid_size: int = 128) -> dict[str, float | int | str]:
    from skimage.measure import perimeter_crofton

    source = sip_root / "code" / "src"
    scripts = sip_root / "code" / "scripts" / "sip_simulation"
    if not source.is_dir() or not scripts.is_dir():
        raise FileNotFoundError(f"SIP source tree not found under {sip_root}")
    for path in (source, scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from compute_dynamic_pore_size import solve_cpu_active_domain

    radius_pixels = math.sqrt(POROSITY / math.pi) * grid_size
    coordinates = np.arange(grid_size, dtype=float) + 0.5 - grid_size / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    pore_2d = xx**2 + yy**2 <= radius_pixels**2
    labels = np.where(np.broadcast_to(pore_2d, (4, grid_size, grid_size)), 1, 2).astype(np.uint8)
    np.save(mask_path, labels)
    pore_area_pixels = int(np.count_nonzero(pore_2d))
    crofton_perimeter_pixels = float(perimeter_crofton(pore_2d, directions=4))
    face_perimeter_pixels = int(
        sum(np.count_nonzero(pore_2d != np.roll(pore_2d, 1, axis=axis)) for axis in (0, 1))
    )
    lambda_crofton_to_radius = 2.0 * pore_area_pixels / (crofton_perimeter_pixels * radius_pixels)
    lambda_voxel_face_to_radius = 2.0 * pore_area_pixels / (face_perimeter_pixels * radius_pixels)
    result = solve_cpu_active_domain(
        labels,
        pore_label=1,
        solid_label=2,
        direction_axis=0,
        voxel_size_m=1.0 / grid_size,
        dtype=np.dtype("complex128"),
        rtol=1e-10,
        maxiter=200,
        use_jacobi=True,
    )
    pore_fraction = float(np.mean(pore_2d))
    effective_unit_conductivity = float(result.effective_conductivity_s_m.real)
    return {
        "grid_shape": f"4x{grid_size}x{grid_size}",
        "pore_fraction": pore_fraction,
        "effective_unit_conductivity": effective_unit_conductivity,
        "formation_factor": 1.0 / effective_unit_conductivity,
        "iterations": int(result.iterations),
        "residual_norm": float(result.residual_norm),
        "relative_parallel_mixing_error": abs(effective_unit_conductivity - pore_fraction) / pore_fraction,
        "lambda_crofton_to_radius": lambda_crofton_to_radius,
        "lambda_voxel_face_to_radius": lambda_voxel_face_to_radius,
        "surface_estimator": "4-direction Crofton correction of the digital circular cross-section; axial field is uniform",
    }


def load_polarization_module(sip_root: Path):
    source = sip_root / "code" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from pore_scale_electrical import polarization

    return polarization


def fit_debye_tau_s(
    frequency_hz: np.ndarray,
    conductivity: np.ndarray,
    sigma0: float,
    sigma_infinity: float,
    epsilon_infinity: float,
    initial_tau_s: float,
) -> float:
    omega = 2.0 * math.pi * frequency_hz

    def prediction(log_tau: np.ndarray) -> np.ndarray:
        tau = math.exp(float(log_tau[0]))
        return sigma_infinity - (sigma_infinity - sigma0) / (1.0 + 1j * omega * tau) + 1j * omega * epsilon_infinity

    real_scale = max(sigma_infinity - sigma0, 1e-15)
    imag_scale = max(float(np.max(np.abs(conductivity.imag - omega * epsilon_infinity))), 1e-15)

    def residual(log_tau: np.ndarray) -> np.ndarray:
        predicted = prediction(log_tau)
        return np.r_[
            (predicted.real - conductivity.real) / real_scale,
            (predicted.imag - conductivity.imag) / imag_scale,
        ]

    fit = least_squares(residual, np.array([math.log(initial_tau_s)]), method="trf")
    if not fit.success:
        raise RuntimeError(f"Debye tau fit failed: {fit.message}")
    return math.exp(float(fit.x[0]))


def simulate_sip(
    radius_um: float,
    lambda_mesh_um: float,
    ac3d: dict[str, float | int | str],
    sip_root: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    polarization = load_polarization_module(sip_root)
    lambda_m = lambda_mesh_um * 1e-6
    radius_m = radius_um * 1e-6
    params = polarization.PolarizationParameters(
        surface_conductance_s=SURFACE_CONDUCTANCE_S,
        diffusion_coefficient_m2_s=ION_DIFFUSION_M2_S,
        dynamic_pore_size_m=lambda_m,
        membrane_polarizability=0.0,
        water_conductivity_s_m=WATER_CONDUCTIVITY_S_M,
        epsilon_0_f_m=EPSILON_0_F_M,
        water_relative_permittivity=WATER_RELATIVE_PERMITTIVITY,
        solid_relative_permittivity=SOLID_RELATIVE_PERMITTIVITY,
    )
    pore_conductance = polarization.pore_polarization_conductance(
        FREQUENCIES_HZ,
        np.array([radius_m]),
        np.array([1.0]),
        params,
    )
    delta_water = polarization.upscale_conductance_to_water_conductivity(pore_conductance, params)
    water = polarization.apparent_water_conductivity(FREQUENCIES_HZ, delta_water, params)
    field_factor = float(ac3d["effective_unit_conductivity"])
    omega = 2.0 * math.pi * FREQUENCIES_HZ
    solid_dielectric = 1j * omega * params.solid_permittivity_f_m
    effective = field_factor * water + (1.0 - field_factor) * solid_dielectric

    tau_s = float(polarization.pore_relaxation_time(np.array([radius_m]), ION_DIFFUSION_M2_S)[0])
    sigma0 = field_factor * WATER_CONDUCTIVITY_S_M
    sigma_infinity = sigma0 + field_factor * 2.0 * SURFACE_CONDUCTANCE_S / lambda_m
    epsilon_infinity = (
        field_factor * params.water_permittivity_f_m
        + (1.0 - field_factor) * params.solid_permittivity_f_m
    )
    paper_equation = (
        sigma_infinity
        - (sigma_infinity - sigma0) / (1.0 + 1j * omega * tau_s)
        + 1j * omega * epsilon_infinity
    )
    equation_error = float(np.max(np.abs(effective - paper_equation)))
    fitted_tau_s = fit_debye_tau_s(
        FREQUENCIES_HZ,
        effective,
        sigma0,
        sigma_infinity,
        epsilon_infinity,
        tau_s,
    )
    frame = pd.DataFrame(
        {
            "radius_um": radius_um,
            "frequency_hz": FREQUENCIES_HZ,
            "sigma_real_s_m": effective.real,
            "sigma_imag_s_m": effective.imag,
            "dielectric_imag_s_m": omega * epsilon_infinity,
            "edl_imag_s_m": effective.imag - omega * epsilon_infinity,
        }
    )
    summary = {
        "tau_analytical_s": radius_um**2 * 1e-12 / (2.0 * ION_DIFFUSION_M2_S),
        "tau_mesh_s": tau_s,
        "tau_fit_s": fitted_tau_s,
        "peak_frequency_hz": 1.0 / (2.0 * math.pi * fitted_tau_s),
        "lambda_sip_crofton_um": lambda_mesh_um,
        "sigma0_s_m": sigma0,
        "sigma_infinity_s_m": sigma_infinity,
        "epsilon_infinity_f_m": epsilon_infinity,
        "paper_eq13_max_abs_error_s_m": equation_error,
    }
    return frame, summary


def second_difference_matrix(size: int) -> np.ndarray:
    return np.diff(np.eye(size), n=2, axis=0)


def joint_problem(
    times_s: np.ndarray,
    nmr_data: np.ndarray,
    frequencies_hz: np.ndarray,
    sip_data: np.ndarray,
    sigma0: float,
    sigma_infinity: float,
    epsilon_infinity: float,
) -> dict[str, np.ndarray | float]:
    radius_m = RADIUS_GRID_UM * 1e-6
    rho_m_s = SURFACE_RELAXIVITY_UM_MS * 1e-3
    t2_s = radius_m / (NMR_SHAPE_FACTOR * rho_m_s)
    tau_s = radius_m**2 / (2.0 * ION_DIFFUSION_M2_S)
    omega = 2.0 * math.pi * frequencies_hz
    nmr_kernel = np.exp(-times_s[:, None] / t2_s[None, :])
    sip_kernel = 1.0 / (1.0 + 1j * omega[:, None] * tau_s[None, :])
    nmr_error = 0.01 * np.maximum(np.abs(nmr_data), 0.02)
    real_error = 0.01 * np.maximum(np.abs(sip_data.real), sigma0)
    edl_imag = np.abs(sip_data.imag - omega * epsilon_infinity)
    imag_error = 0.35 * np.maximum(edl_imag, max(float(edl_imag.max()) * 0.02, 1e-15))
    return {
        "nmr_kernel": nmr_kernel,
        "sip_kernel": sip_kernel,
        "nmr_data": nmr_data,
        "sip_data": sip_data,
        "nmr_error": nmr_error,
        "real_error": real_error,
        "imag_error": imag_error,
        "omega": omega,
        "sigma0": sigma0,
        "sigma_infinity": sigma_infinity,
        "epsilon_infinity": epsilon_infinity,
        "regularizer": second_difference_matrix(len(RADIUS_GRID_UM)),
    }


def fit_joint_distribution(
    problem: dict[str, np.ndarray | float],
    beta: float,
    regularization: float,
    initial: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    n_bins = len(RADIUS_GRID_UM)
    reference = np.full(n_bins, math.log(1.0 / n_bins))
    x0 = reference if initial is None else np.asarray(initial, dtype=float)
    nmr_kernel = np.asarray(problem["nmr_kernel"])
    sip_kernel = np.asarray(problem["sip_kernel"])
    nmr_data = np.asarray(problem["nmr_data"])
    sip_data = np.asarray(problem["sip_data"])
    nmr_error = np.asarray(problem["nmr_error"])
    real_error = np.asarray(problem["real_error"])
    imag_error = np.asarray(problem["imag_error"])
    omega = np.asarray(problem["omega"])
    sigma0 = float(problem["sigma0"])
    sigma_infinity = float(problem["sigma_infinity"])
    epsilon_infinity = float(problem["epsilon_infinity"])
    regularizer = np.asarray(problem["regularizer"])
    dispersion = sigma_infinity - sigma0

    def calculate(log_weights: np.ndarray):
        weights = np.exp(log_weights)
        nmr_prediction = nmr_kernel @ weights
        sip_prediction = (
            sigma_infinity
            - dispersion * (sip_kernel @ weights)
            + 1j * omega * epsilon_infinity
        )
        return weights, nmr_prediction, sip_prediction

    def residual(log_weights: np.ndarray) -> np.ndarray:
        _, nmr_prediction, sip_prediction = calculate(log_weights)
        return np.r_[
            beta * (sip_prediction.real - sip_data.real) / real_error,
            beta * (sip_prediction.imag - sip_data.imag) / imag_error,
            (nmr_prediction - nmr_data) / nmr_error,
            math.sqrt(regularization) * (regularizer @ (log_weights - reference)),
        ]

    def jacobian(log_weights: np.ndarray) -> np.ndarray:
        weights = np.exp(log_weights)
        nmr_jacobian = nmr_kernel * weights[None, :]
        sip_jacobian = -dispersion * sip_kernel * weights[None, :]
        return np.vstack(
            [
                beta * sip_jacobian.real / real_error[:, None],
                beta * sip_jacobian.imag / imag_error[:, None],
                nmr_jacobian / nmr_error[:, None],
                math.sqrt(regularization) * regularizer,
            ]
        )

    fit = least_squares(
        residual,
        x0,
        jac=jacobian,
        bounds=(math.log(1e-12), 0.0),
        method="trf",
        max_nfev=1000,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Joint inversion returned non-finite parameters: {fit.message}")
    weights, nmr_prediction, sip_prediction = calculate(fit.x)
    unweighted = np.r_[
        (sip_prediction.real - sip_data.real) / real_error,
        (sip_prediction.imag - sip_data.imag) / imag_error,
        (nmr_prediction - nmr_data) / nmr_error,
    ]
    return {
        "log_weights": fit.x,
        "weights": weights,
        "nmr_prediction": nmr_prediction,
        "sip_prediction": sip_prediction,
        "data_misfit": float(np.mean(unweighted**2)),
        "weighted_data_misfit": float(np.mean(residual(fit.x)[: unweighted.size] ** 2)),
        "roughness": float(np.linalg.norm(regularizer @ (fit.x - reference))),
        "iterations": int(fit.nfev),
        "converged": bool(fit.success),
        "termination": str(fit.message),
    }


def two_step_joint_inversion(problem: dict[str, np.ndarray | float]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    beta_rows = []
    beta_results = []
    for beta in BETA_GRID:
        result = fit_joint_distribution(problem, float(beta), INTERMEDIATE_LAMBDA)
        beta_results.append(result)
        beta_rows.append(
            {
                "beta": beta,
                "lambda": INTERMEDIATE_LAMBDA,
                "data_misfit": result["data_misfit"],
                "weighted_data_misfit": result["weighted_data_misfit"],
                "roughness": result["roughness"],
                "iterations": result["iterations"],
                "converged": result["converged"],
                "termination": result["termination"],
            }
        )
    beta_candidates = [index for index, row in enumerate(beta_rows) if row["converged"]]
    if not beta_candidates:
        raise RuntimeError("No beta-scan candidate converged")
    beta_index = min(beta_candidates, key=lambda index: beta_rows[index]["data_misfit"])
    selected_beta = float(BETA_GRID[beta_index])
    beta_initial = np.asarray(beta_results[beta_index]["log_weights"])

    lambda_rows = []
    lambda_results = []
    lambda_initial = beta_initial
    uniform_initial = np.full(len(RADIUS_GRID_UM), math.log(1.0 / len(RADIUS_GRID_UM)))
    for value in LAMBDA_GRID:
        initial = uniform_initial if value > INTERMEDIATE_LAMBDA else lambda_initial
        result = fit_joint_distribution(problem, selected_beta, float(value), initial=initial)
        lambda_initial = np.asarray(result["log_weights"])
        lambda_results.append(result)
        lambda_rows.append(
            {
                "beta": selected_beta,
                "lambda": value,
                "data_misfit": result["data_misfit"],
                "weighted_data_misfit": result["weighted_data_misfit"],
                "roughness": result["roughness"],
                "iterations": result["iterations"],
                "converged": result["converged"],
                "termination": result["termination"],
            }
        )
    lambda_candidates = [index for index, row in enumerate(lambda_rows) if row["converged"]]
    if not lambda_candidates:
        raise RuntimeError("No lambda-scan candidate converged")
    lambda_index = min(lambda_candidates, key=lambda index: lambda_rows[index]["data_misfit"])
    selected = lambda_results[lambda_index]
    selected["beta"] = selected_beta
    selected["lambda"] = float(LAMBDA_GRID[lambda_index])
    return selected, pd.DataFrame(beta_rows), pd.DataFrame(lambda_rows)


def reduced_nmr_data(frame: pd.DataFrame, count: int = 50) -> tuple[np.ndarray, np.ndarray]:
    indices = np.unique(np.r_[0, np.geomspace(1, len(frame) - 1, count - 1).astype(int)])
    return frame["time_ms"].to_numpy()[indices] * 1e-3, frame["signal_normalized"].to_numpy()[indices]


def save_validation_figure(
    output_path: Path,
    summary: pd.DataFrame,
    nmr_data: pd.DataFrame,
    sip_data: pd.DataFrame,
    distributions: pd.DataFrame,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    colors = ["#3B6FB6", "#5A9E6F", "#D28E38", "#A64B4B"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)

    ax = axes[0, 0]
    side = math.sqrt(math.pi / POROSITY)
    ax.add_patch(Rectangle((-side / 2, -side / 2), side, side, facecolor="#E3E3E3", edgecolor="#4A4A4A"))
    ax.add_patch(Circle((0, 0), 1.0, facecolor="#8EC6E8", edgecolor="#2C6485", lw=1.2))
    ax.text(0, 0, r"$\odot$", ha="center", va="center", fontsize=22, color="#163D58")
    ax.text(0, -0.35, "axial E", ha="center", va="center", fontsize=7)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.8, 1.8), aspect="equal")
    ax.axis("off")
    ax.set_title("a  Shared straight-cylinder geometry", loc="left", weight="bold")
    ax.text(0.02, 0.02, r"$\phi=0.50$, $L=5a$", transform=ax.transAxes)

    ax = axes[0, 1]
    for color, radius in zip(colors, RADII_UM):
        group = nmr_data[nmr_data["radius_um"] == radius]
        ax.plot(group["time_ms"] / 1000.0, group["signal_normalized"], color=color, lw=1.4, label=f"{radius:g} µm")
    ax.set(xlabel="Time (s)", ylabel="Normalized $M(t)$", yscale="log", ylim=(1e-3, 1.05))
    ax.grid(alpha=0.18, which="both")
    ax.legend(ncol=2, fontsize=6)
    ax.set_title("b  NMR Bloch-Torrey decay", loc="left", weight="bold")

    ax = axes[0, 2]
    for color, radius in zip(colors, RADII_UM):
        group = sip_data[sip_data["radius_um"] == radius]
        amplitude = group["edl_imag_s_m"].to_numpy()
        ax.plot(group["frequency_hz"], amplitude / amplitude.max(), color=color, lw=1.4, label=f"{radius:g} µm")
    ax.set(xlabel="Frequency (Hz)", ylabel=r"Normalized EDL $\sigma''$", xscale="log", ylim=(-0.02, 1.05))
    ax.grid(alpha=0.18, which="both")
    ax.set_title("c  SIP pore polarization", loc="left", weight="bold")

    ax = axes[1, 0]
    t2_s = summary["t2_fit_ms"].to_numpy() * 1e-3
    tau_s = summary["tau_fit_s"].to_numpy()
    slope, intercept = np.polyfit(np.log10(t2_s), np.log10(tau_s), 1)
    xline = np.geomspace(t2_s.min() * 0.85, t2_s.max() * 1.15, 100)
    theory = 2.0 * (SURFACE_RELAXIVITY_UM_MS * 1e-3) ** 2 / ION_DIFFUSION_M2_S * xline**2
    ax.plot(xline, theory, color="#303030", ls="--", lw=1.2, label="Fast-diffusion theory")
    ax.scatter(t2_s, tau_s, c=colors, s=28, zorder=3, label=f"Numerical, slope={slope:.3f}")
    ax.set(xlabel=r"$T_2$ (s)", ylabel=r"$\tau$ (s)", xscale="log", yscale="log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.18, which="both")
    ax.legend(fontsize=6)
    ax.set_title("d  Shared-radius scaling", loc="left", weight="bold")

    ax = axes[1, 1]
    for color, radius in zip(colors, RADII_UM):
        group = distributions[distributions["radius_true_um"] == radius]
        normalized = group["joint_weight"].to_numpy(copy=True)
        normalized /= normalized.max()
        ax.plot(group["radius_bin_um"], normalized, color=color, lw=1.4, label=f"{radius:g} µm")
        ax.axvline(radius, color=color, ls=":", lw=0.8, alpha=0.7)
    ax.set(xlabel="Pore radius (µm)", ylabel="Recovered relative volume / peak", xscale="log", ylim=(-0.02, 1.05))
    ax.grid(alpha=0.18, which="both")
    ax.set_title("e  Joint NMR-SIP inversion", loc="left", weight="bold")

    ax = axes[1, 2]
    metrics = [
        ("NMR $T_2$", "t2_relative_error", "#3B6FB6"),
        (r"SIP $\tau$", "tau_relative_error", "#D28E38"),
        (r"Crofton $\Lambda$", "lambda_relative_error", "#5A9E6F"),
        ("Joint $R$", "joint_radius_relative_error", "#A64B4B"),
    ]
    x = np.arange(len(summary))
    width = 0.19
    for offset, (label, column, color) in enumerate(metrics):
        values = summary[column].to_numpy() * 100.0
        ax.bar(x + (offset - 1.5) * width, values, width=width, color=color, label=label)
    ax.set(xticks=x, xticklabels=[f"{value:g}" for value in summary["radius_true_um"]], xlabel="True radius (µm)", ylabel="Relative error (%)", ylim=(0.0, 0.65))
    ax.grid(alpha=0.18, axis="y")
    ax.legend(fontsize=5.7, ncol=2)
    ax.set_title("f  Numerical validation", loc="left", weight="bold")

    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, slope: float, all_passed: bool, ac3d: dict) -> None:
    lambda_at_lower_bound = int((summary["selected_lambda"] == LAMBDA_GRID.min()).sum())
    columns = [
        "radius_true_um",
        "t2_fit_ms",
        "t2_exact_ms",
        "tau_fit_s",
        "tau_analytical_s",
        "lambda_sip_crofton_um",
        "joint_peak_radius_um",
        "all_case_checks_passed",
    ]
    table = summary[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{value:.8g}" if isinstance(value, float) else str(value) for value in values) + " |")
    markdown_table = "\n".join([header, separator, *rows])
    path.write_text(
        "# 直圆柱理想几何 NMR-SIP 联合验证\n\n"
        f"结论：{'全部预设验收项通过。' if all_passed else '至少一项未通过，不能判定 same-r 基准闭合。'} "
        f"四个模型的数值关系为 `log10(tau) = {slope:.6f} log10(T2) + const`。\n\n"
        "## 模型与求解\n\n"
        "模型为孔隙率 0.50 的绝缘方柱及轴向贯通圆柱孔，半径为 2、5、10、20 µm，长度定义为 5a。"
        "NMR 使用圆截面二维 Robin-Bloch-Torrey 有限元求解；SIP 使用同一挤出圆柱的 4 层三维轴向周期代表单胞求解场。"
        "由于几何沿轴向不变，该单胞给出与设计长度 L=5a 相同的轴向有效电导率，但未显式离散整根 L=5a 圆柱。"
        "SIP 的动态孔径用体素截面的四方向 Crofton 周长修正估计；原始 voxel-face 表面积估计也保留，"
        "但因阶梯边界偏差不能用于 2% 的圆柱几何验收。仅启用 Niu 孔/EDL Debye 极化，"
        "膜极化、界面极化、体弛豫和噪声均关闭。\n\n"
        "联合反演直接使用 NMR 衰减、SIP 实部和虚部，而不是先反演出的 T2 或 tau。"
        "前向核采用 Niu & Zhang (2018) Eqs. (1), (4), (7), (13)；圆柱形状因子为 alpha=2。"
        "孔径权重以 log(f_i) 参数化，使用二阶 Tikhonov 正则和解析 Jacobian 的信赖域 Gauss-Newton 求解，"
        "并按论文的两步方式先扫描 beta、再扫描 lambda。\n\n"
        "## 数值结果\n\n"
        + markdown_table
        + "\n\n"
        "## 解释边界\n\n"
        f"- SIP 截面原始 voxel-face 积分给出 Lambda/R={float(ac3d['lambda_voxel_face_to_radius']):.6f}"
        f"（误差 {100.0 * abs(float(ac3d['lambda_voxel_face_to_radius']) - 1.0):.2f}%）；"
        f"四方向 Crofton 修正给出 {float(ac3d['lambda_crofton_to_radius']):.6f}"
        f"（误差 {100.0 * abs(float(ac3d['lambda_crofton_to_radius']) - 1.0):.3f}%）。"
        "本次 Lambda 验收采用后者。\n"
        "- 这是二维圆截面 NMR 对应三维直圆柱 SIP 的受控共同几何基准，不是真实三维 NMR 求解。\n"
        "- SIP 的 tau=r^2/(2D+) 是预设的论文孤立 EDL 核；拟合回 tau 是核的一致性检查，"
        "不是从独立表面扩散 PDE 发现该系数。Crofton Lambda 是几何估计，不是现有 voxel-face 场积分的原始输出。\n"
        "- 论文未给出孔径箱、参考模型、扫描节点、停止阈值和随机种子；本结果复现其方程与求解结构，"
        "不宣称逐数值复现论文图 3-5。\n"
        f"- 本轮为无噪声 Case-1 型基准；{lambda_at_lower_bound}/4 组的选中 lambda 位于扫描下界，"
        "因此不能据此声称正则化参数稳定。仅收敛候选参与选择，逐点状态见 lambda_scan.csv。"
        "含噪声、体弛豫、有限扩散、膜极化和孔体-孔喉几何应分别作为后续敏感性试验。\n",
        encoding="utf-8",
    )


def run(output_dir: Path, sip_root: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    figure_dir.mkdir(exist_ok=True)
    source_dir.mkdir(exist_ok=True)

    mask_path = source_dir / "ideal_cylinder_sip_labels_4x128x128.npy"
    ac3d = run_ac3d_unit_field(sip_root, mask_path)
    nmr_frames = []
    sip_frames = []
    distribution_frames = []
    beta_frames = []
    lambda_frames = []
    summary_rows = []
    mesh_paths = []

    for radius_um in RADII_UM:
        print(f"Running R={radius_um:g} um", flush=True)
        mesh_path = source_dir / f"ideal_cylinder_R{radius_um:g}um_nmr_mesh.bms"
        nmr_frame, nmr_summary = simulate_nmr(float(radius_um), mesh_path)
        mesh_paths.append(mesh_path)
        lambda_sip_um = float(ac3d["lambda_crofton_to_radius"]) * radius_um
        sip_frame, sip_summary = simulate_sip(
            float(radius_um),
            lambda_sip_um,
            ac3d,
            sip_root,
        )
        times_s, nmr_reduced = reduced_nmr_data(nmr_frame)
        sip_complex = sip_frame["sigma_real_s_m"].to_numpy() + 1j * sip_frame["sigma_imag_s_m"].to_numpy()
        problem = joint_problem(
            times_s,
            nmr_reduced,
            sip_frame["frequency_hz"].to_numpy(),
            sip_complex,
            sip_summary["sigma0_s_m"],
            sip_summary["sigma_infinity_s_m"],
            sip_summary["epsilon_infinity_f_m"],
        )
        joint, beta_scan, lambda_scan = two_step_joint_inversion(problem)
        weights = np.asarray(joint["weights"])
        peak_radius_um = float(RADIUS_GRID_UM[int(np.argmax(weights))])

        distribution_frames.append(
            pd.DataFrame(
                {
                    "radius_true_um": radius_um,
                    "radius_bin_um": RADIUS_GRID_UM,
                    "joint_weight": weights,
                    "joint_weight_normalized_sum": weights / weights.sum(),
                }
            )
        )
        beta_scan.insert(0, "radius_true_um", radius_um)
        lambda_scan.insert(0, "radius_true_um", radius_um)
        beta_scan["selected"] = beta_scan["beta"] == float(joint["beta"])
        lambda_scan["selected"] = lambda_scan["lambda"] == float(joint["lambda"])
        beta_frames.append(beta_scan)
        lambda_frames.append(lambda_scan)

        t2_error = abs(float(nmr_summary["t2_fit_ms"]) - float(nmr_summary["t2_exact_ms"])) / float(nmr_summary["t2_exact_ms"])
        tau_error = abs(sip_summary["tau_fit_s"] - sip_summary["tau_analytical_s"]) / sip_summary["tau_analytical_s"]
        lambda_error = abs(lambda_sip_um - radius_um) / radius_um
        nmr_mesh_lambda_error = abs(float(nmr_summary["lambda_mesh_um"]) - radius_um) / radius_um
        joint_error = abs(peak_radius_um - radius_um) / radius_um
        case_passed = t2_error < 0.02 and tau_error < 0.05 and lambda_error < 0.02 and joint_error < 0.05
        summary_rows.append(
            {
                "radius_true_um": radius_um,
                "cell_side_um": radius_um * math.sqrt(math.pi / POROSITY),
                "cylinder_length_um": LENGTH_FACTOR * radius_um * math.sqrt(math.pi / POROSITY),
                "sip_cross_section_voxel_size_um": radius_um * math.sqrt(math.pi / POROSITY) / 128.0,
                "sip_periodic_axial_cell_length_um": 4.0 * radius_um * math.sqrt(math.pi / POROSITY) / 128.0,
                **nmr_summary,
                **sip_summary,
                "t2_fast_diffusion_ms": radius_um / (NMR_SHAPE_FACTOR * SURFACE_RELAXIVITY_UM_MS),
                "t2_relative_error": t2_error,
                "tau_relative_error": tau_error,
                "lambda_relative_error": lambda_error,
                "nmr_mesh_lambda_relative_error": nmr_mesh_lambda_error,
                "lambda_sip_voxel_face_um": float(ac3d["lambda_voxel_face_to_radius"]) * radius_um,
                "lambda_sip_voxel_face_relative_error": abs(float(ac3d["lambda_voxel_face_to_radius"]) - 1.0),
                "joint_peak_radius_um": peak_radius_um,
                "joint_radius_relative_error": joint_error,
                "joint_weight_sum": float(weights.sum()),
                "selected_beta": float(joint["beta"]),
                "selected_lambda": float(joint["lambda"]),
                "joint_data_misfit": float(joint["data_misfit"]),
                "all_case_checks_passed": case_passed,
            }
        )
        nmr_frames.append(nmr_frame)
        sip_frames.append(sip_frame)
        print(
            f"Completed R={radius_um:g} um: T2 error={t2_error:.3%}, "
            f"tau error={tau_error:.3%}, joint radius error={joint_error:.3%}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    nmr_data = pd.concat(nmr_frames, ignore_index=True)
    sip_data = pd.concat(sip_frames, ignore_index=True)
    distributions = pd.concat(distribution_frames, ignore_index=True)
    beta_scans = pd.concat(beta_frames, ignore_index=True)
    lambda_scans = pd.concat(lambda_frames, ignore_index=True)
    slope, intercept = np.polyfit(
        np.log10(summary["t2_fit_ms"].to_numpy() * 1e-3),
        np.log10(summary["tau_fit_s"].to_numpy()),
        1,
    )
    formation_factor_error = abs(float(ac3d["formation_factor"]) - 2.0) / 2.0
    all_passed = bool(
        summary["all_case_checks_passed"].all()
        and formation_factor_error < 0.02
        and abs(slope - 2.0) < 0.05
        and int(ac3d["iterations"]) >= 0
        and float(ac3d["residual_norm"]) < 1e-8
        and float(summary["paper_eq13_max_abs_error_s_m"].max()) < 1e-12
    )

    files = {
        "summary_csv": source_dir / "ideal_cylinder_validation_summary.csv",
        "nmr_csv": source_dir / "ideal_cylinder_nmr_decay.csv",
        "sip_csv": source_dir / "ideal_cylinder_sip_spectra.csv",
        "joint_csv": source_dir / "ideal_cylinder_joint_psd.csv",
        "beta_csv": source_dir / "ideal_cylinder_beta_scan.csv",
        "lambda_csv": source_dir / "ideal_cylinder_lambda_scan.csv",
        "figure_png": figure_dir / "ideal_cylinder_nmr_sip_joint_validation.png",
        "report_md": output_dir / "validation_report.md",
        "manifest_json": output_dir / "run_manifest.json",
    }
    for frame, key in [
        (summary, "summary_csv"),
        (nmr_data, "nmr_csv"),
        (sip_data, "sip_csv"),
        (distributions, "joint_csv"),
        (beta_scans, "beta_csv"),
        (lambda_scans, "lambda_csv"),
    ]:
        frame.to_csv(files[key], index=False, encoding="utf-8-sig")
    save_validation_figure(files["figure_png"], summary, nmr_data, sip_data, distributions)
    write_report(files["report_md"], summary, float(slope), all_passed, ac3d)

    script_path = Path(__file__).resolve()
    manifest = {
        "status": "passed" if all_passed else "failed",
        "model_dimension": "2D circular-cross-section NMR mapped to the same extruded 3D straight-cylinder SIP geometry",
        "geometry": {
            "radii_um": RADII_UM.tolist(),
            "porosity": POROSITY,
            "cell_side_relation": "a = R*sqrt(pi/phi)",
            "length_relation": "Physical target L=5a. SIP solves a 4-voxel axial periodic representative cell rather than explicitly meshing all 5a; this is exact for the translationally invariant axial response.",
            "shared_scale": "r_NMR = r_SIP = Lambda = R",
            "sip_labels_npy": str(mask_path.resolve()),
            "sip_phase_labels": {"pore": 1, "insulating_solid": 2},
            "nmr_mesh_bms": [str(path.resolve()) for path in mesh_paths],
        },
        "nmr": {
            "solver": "existing pyGIMLi 2D Robin-Bloch-Torrey finite-element stepper",
            "diffusion_um2_ms": NMR_DIFFUSION_UM2_MS,
            "surface_relaxivity_um_ms": SURFACE_RELAXIVITY_UM_MS,
            "bulk_t2_ms": None,
            "time_step_ms": NMR_DT_MS,
            "time_max_ms": NMR_MAX_TIME_MS,
            "shape_factor_alpha": NMR_SHAPE_FACTOR,
        },
        "sip": {
            "solver": "existing AC3D active-domain axial field plus existing Niu pore-polarization kernel",
            "frequencies_hz": [float(FREQUENCIES_HZ.min()), float(FREQUENCIES_HZ.max())],
            "frequency_count": len(FREQUENCIES_HZ),
            "ion_diffusion_m2_s": ION_DIFFUSION_M2_S,
            "surface_conductance_s": SURFACE_CONDUCTANCE_S,
            "water_conductivity_s_m": WATER_CONDUCTIVITY_S_M,
            "membrane_polarization": False,
            "interfacial_polarization": False,
            "ac3d": ac3d,
            "tau_origin": "Niu pore-polarization kernel imposed from the nominal radius; Debye refit is a consistency test, not an independent Stern-layer diffusion solve",
            "formation_factor_relative_error_vs_2": formation_factor_error,
        },
        "joint_inversion": {
            "paper": "Niu and Zhang (2018), GJI 212, 1791-1805, doi:10.1093/gji/ggx501",
            "equations": [1, 4, 7, 13, 14, 15, 16, 17, 18, 19, 20],
            "known_fixed_parameters": ["D_plus", "rho_2", "sigma_0", "sigma_infinity", "epsilon_infinity", "E_0"],
            "unknown": "log pore-radius relative weights",
            "radius_grid_um": [float(RADIUS_GRID_UM.min()), float(RADIUS_GRID_UM.max()), len(RADIUS_GRID_UM)],
            "solver": "analytic-Jacobian trust-region Gauss-Newton least squares",
            "regularizer": "second row difference of identity on log weights",
            "beta_grid": BETA_GRID.tolist(),
            "lambda_grid": LAMBDA_GRID.tolist(),
            "intermediate_lambda": INTERMEDIATE_LAMBDA,
            "selection": "minimum unweighted standardized data misfit, beta first then lambda",
            "noise_added": False,
            "nominal_weights": "paper-like 1% NMR, 1% SIP real, 35% SIP imaginary with finite floors",
        },
        "validation": {
            "log_tau_vs_log_t2_slope": float(slope),
            "log_tau_vs_log_t2_intercept": float(intercept),
            "criteria": {
                "nmr_t2_relative_error_lt": 0.02,
                "sip_tau_relative_error_lt": 0.05,
                "lambda_relative_error_lt": 0.02,
                "joint_peak_radius_relative_error_lt": 0.05,
                "formation_factor_relative_error_lt": 0.02,
                "slope_absolute_error_lt": 0.05,
                "ac3d_true_residual_lt": 1e-8,
                "sip_paper_eq13_max_abs_error_s_m_lt": 1e-12,
            },
            "all_passed": all_passed,
        },
        "review_boundaries": [
            "The test validates the imposed Niu-Zhang shared-radius mapping; it does not independently derive the EDL coefficient for arbitrary cylindrical pores.",
            "The NMR solve is 2D and the SIP geometry is its translationally invariant 3D extrusion.",
            "Bulk relaxation, membrane polarization, interfacial polarization, and noise are excluded from this first benchmark.",
            "Paper implementation details omitted by the article prevent bitwise reproduction of its figures.",
        ],
        "provenance": {
            "python": sys.version,
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "nmr_repo_commit": git_commit(script_path.parents[1]),
            "sip_repo_commit": git_commit(sip_root),
        },
        "outputs": {key: str(value.resolve()) for key, value in files.items()},
    }
    files["manifest_json"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def self_check() -> None:
    expected_ms = np.array([200.3, 501.6, 1006.3, 2025.1])
    calculated = np.array([bessel_t2_ms(float(radius))[0] for radius in RADII_UM])
    assert np.allclose(calculated, expected_ms, rtol=2e-3)
    assert np.allclose(
        RADII_UM**2 * 1e-12 / (2.0 * ION_DIFFUSION_M2_S),
        np.array([0.0015384615, 0.0096153846, 0.0384615385, 0.1538461538]),
        rtol=1e-7,
    )
    matrix = second_difference_matrix(7)
    assert matrix.shape == (5, 7)
    assert np.allclose(matrix @ np.ones(7), 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_outputs/ideal_cylinder_nmr_sip_joint_20260919"),
    )
    parser.add_argument(
        "--sip-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "sip模拟",
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    self_check()
    if args.self_check:
        print("self-check passed")
        return
    manifest = run(args.output_dir, args.sip_root)
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
