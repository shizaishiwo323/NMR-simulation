"""PDE simulation for a segmented 2D three-phase image.

The image is expected to contain:
- 255: solid grains
- 102: water
- 153: air
- 0: outside/tube region ignored by the simulation

This script uses the original image resolution by default. It does not
generate new saturation states from Young-Laplace/minimum-energy interfaces.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.optimize import nnls
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import factorized

from simulation_control import SETTINGS


def cfg_get(path, default=None):
    value = SETTINGS
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


D_UM2_PER_MS = 2.0
T2B_MS = 3000.0
RHO_UM_PER_MS = 0.005


def downsample_nearest(labels: np.ndarray, max_size: int | None) -> tuple[np.ndarray, float]:
    if max_size is None:
        return labels, 1.0
    scale = max(labels.shape) / float(max_size)
    if scale <= 1.0:
        return labels, 1.0
    zoom = 1.0 / scale
    small = ndimage.zoom(labels, zoom=zoom, order=0)
    return small.astype(labels.dtype), scale


def classify_and_clean(raw: np.ndarray, settings: dict) -> np.ndarray:
    solid = int(settings["solid_value"])
    water = int(settings["water_value"])
    air = int(settings["air_value"])
    outside = int(settings["outside_value"])
    valid = np.array([outside, water, air, solid], dtype=np.int16)
    cleaned = raw.astype(np.int16).copy()
    for value in np.unique(cleaned):
        if value in valid:
            continue
        distances = np.abs(valid - int(value))
        cleaned[cleaned == value] = int(valid[int(np.argmin(distances))])

    sigma = float(settings.get("interface_sigma_px", 0.0))
    if sigma > 0:
        phase_masks = {
            outside: cleaned == outside,
            solid: cleaned == solid,
            water: cleaned == water,
            air: cleaned == air,
        }
        # Keep outside fixed; smooth the three internal phase probabilities.
        internal_values = [solid, water, air]
        scores = []
        for value in internal_values:
            score = ndimage.gaussian_filter(phase_masks[value].astype(float), sigma=sigma, mode="nearest")
            scores.append(score)
        winner = np.argmax(np.stack(scores, axis=0), axis=0)
        smoothed = np.asarray(internal_values, dtype=np.uint8)[winner]
        smoothed[phase_masks[outside]] = outside
        cleaned = smoothed.astype(np.int16)

    radius = int(settings.get("smooth_radius_px", 0))
    if radius > 0:
        water_mask = cleaned == water
        solid_mask = cleaned == solid
        outside_mask = cleaned == outside
        air_mask = cleaned == air

        # Remove single-pixel salt/holes without changing the large phase topology.
        water_mask = ndimage.binary_opening(water_mask, structure=np.ones((radius + 1, radius + 1)))
        water_mask = ndimage.binary_closing(water_mask, structure=np.ones((radius + 1, radius + 1)))
        solid_mask = ndimage.binary_closing(solid_mask, structure=np.ones((radius + 1, radius + 1)))

        cleaned2 = np.full(cleaned.shape, air, dtype=np.int16)
        cleaned2[outside_mask] = outside
        cleaned2[solid_mask] = solid
        cleaned2[water_mask & ~solid_mask & ~outside_mask] = water
        cleaned2[air_mask & ~solid_mask & ~outside_mask & ~water_mask] = air
        cleaned = cleaned2

    return cleaned.astype(np.uint8)


def save_phase_preview(labels: np.ndarray, path: Path, settings: dict) -> None:
    solid = int(settings["solid_value"])
    water = int(settings["water_value"])
    air = int(settings["air_value"])
    outside = int(settings["outside_value"])
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    rgb[labels == outside] = [0, 0, 0]
    rgb[labels == air] = [230, 230, 230]
    rgb[labels == water] = [40, 120, 220]
    rgb[labels == solid] = [120, 120, 120]
    Image.fromarray(rgb).save(path)


def build_water_operator(labels: np.ndarray, pixel_size_um: float, settings: dict):
    water_value = int(settings["water_value"])
    solid_value = int(settings["solid_value"])
    outside_value = int(settings["outside_value"])
    water = labels == water_value
    idx = -np.ones(labels.shape, dtype=int)
    coords = np.argwhere(water)
    idx[water] = np.arange(coords.shape[0])
    n = coords.shape[0]
    if n == 0:
        raise ValueError("No water pixels found in the cleaned image.")

    rows = []
    cols = []
    data = []
    boundary_counts = np.zeros(n, dtype=float)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for k, (r, c) in enumerate(coords):
        degree = 0
        boundary = 0
        for dr, dc in neighbor_offsets:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= labels.shape[0] or cc >= labels.shape[1]:
                boundary += 1
                continue
            if water[rr, cc]:
                rows.append(k)
                cols.append(idx[rr, cc])
                data.append(-1.0)
                degree += 1
            elif labels[rr, cc] in (solid_value, outside_value):
                boundary += 1
        rows.append(k)
        cols.append(k)
        data.append(float(degree))
        boundary_counts[k] = boundary

    graph_laplacian = csr_matrix((data, (rows, cols)), shape=(n, n))
    solid_sink = boundary_counts / max(pixel_size_um, 1e-12)
    return coords, graph_laplacian, solid_sink


def build_decay_solver(graph_laplacian, solid_sink, dt_ms, pixel_size_um):
    n = graph_laplacian.shape[0]
    lhs = (
        eye(n, format="csr")
        + graph_laplacian * (D_UM2_PER_MS * dt_ms / max(pixel_size_um**2, 1e-12))
        + eye(n, format="csr") * (dt_ms / T2B_MS)
        + diags(RHO_UM_PER_MS * dt_ms * solid_sink, format="csr")
    )
    return factorized(lhs.tocsc())


def solve_decay(graph_laplacian, solid_sink, times_ms, dt_ms, pixel_size_um, pixel_area_um2):
    solve = build_decay_solver(graph_laplacian, solid_sink, dt_ms, pixel_size_um)
    n = graph_laplacian.shape[0]
    u = np.ones(n, dtype=float)
    amps = []
    for _ in times_ms:
        amps.append(float(np.sum(u) * pixel_area_um2))
        u = solve(u)
    return np.asarray(amps)


def solve_uncoupled_component_decay(labels, components, n_components, times_ms, dt_ms, pixel_size_um, pixel_area_um2, settings):
    """Solve each connected water component independently and sum signals."""

    water_value = int(settings["water_value"])
    total = np.zeros_like(times_ms, dtype=float)
    kept = 0
    for label in range(1, n_components + 1):
        comp_mask = components == label
        if int(np.sum(comp_mask)) < 3:
            continue
        comp_labels = np.full(labels.shape, int(settings["air_value"]), dtype=np.uint8)
        comp_labels[labels == int(settings["solid_value"])] = int(settings["solid_value"])
        comp_labels[labels == int(settings["outside_value"])] = int(settings["outside_value"])
        comp_labels[comp_mask] = water_value
        _, lap, sink = build_water_operator(comp_labels, pixel_size_um=pixel_size_um, settings=settings)
        total += solve_decay(lap, sink, times_ms, dt_ms, pixel_size_um, pixel_area_um2)
        kept += 1
    return total, kept


def invert_t2(times_ms, signal, t2_bins_ms, alpha):
    if np.max(signal) <= 1e-12:
        return np.zeros_like(t2_bins_ms)
    norm = signal / signal[0]
    kernel = np.exp(-np.outer(times_ms, 1.0 / t2_bins_ms))
    n = len(t2_bins_ms)
    reg = np.zeros((n, n))
    for i in range(1, n - 1):
        reg[i, i - 1] = 1
        reg[i, i] = -2
        reg[i, i + 1] = 1
    reg[0, 0] = -1
    reg[0, 1] = 1
    reg[-1, -2] = 1
    reg[-1, -1] = -1
    spec, _ = nnls(np.vstack([kernel, alpha * reg]), np.concatenate([norm, np.zeros(n)]))
    return spec * signal[0]


def invert_t2_auto_lcurve(times_ms, signal, t2_bins_ms):
    """Automatic alpha selection using L-curve curvature on fixed NNLS grids."""

    if np.max(signal) <= 1e-12:
        return np.zeros_like(t2_bins_ms), np.nan, pd.DataFrame()
    norm = signal / signal[0]
    kernel = np.exp(-np.outer(times_ms, 1.0 / t2_bins_ms))
    n = len(t2_bins_ms)
    reg = np.zeros((n, n))
    for i in range(1, n - 1):
        reg[i, i - 1] = 1
        reg[i, i] = -2
        reg[i, i + 1] = 1
    reg[0, 0] = -1
    reg[0, 1] = 1
    reg[-1, -2] = 1
    reg[-1, -1] = -1

    alphas = np.logspace(-5, 2, 36)
    spectra = []
    residuals = []
    roughness = []
    for alpha in alphas:
        spec, _ = nnls(np.vstack([kernel, alpha * reg]), np.concatenate([norm, np.zeros(n)]))
        spectra.append(spec)
        residuals.append(float(np.linalg.norm(kernel @ spec - norm)))
        roughness.append(float(np.linalg.norm(reg @ spec)))

    x = np.log10(np.maximum(residuals, 1e-30))
    y = np.log10(np.maximum(roughness, 1e-30))
    curvature = np.zeros_like(alphas)
    for i in range(1, len(alphas) - 1):
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i + 1], y[i + 1]
        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        side = math.dist((x1, y1), (x3, y3)) * math.dist((x1, y1), (x2, y2)) * math.dist((x2, y2), (x3, y3))
        curvature[i] = 2.0 * area2 / side if side > 0 else 0.0
    best = int(np.argmax(curvature))
    metrics = pd.DataFrame(
        {
            "alpha": alphas,
            "residual_norm": residuals,
            "roughness_norm": roughness,
            "curvature": curvature,
            "is_best": np.arange(len(alphas)) == best,
        }
    )
    return spectra[best] * signal[0], float(alphas[best]), metrics


def invert_2d_t2(signal_2d, t1_axis, t2_axis, bins, alpha):
    a1 = np.exp(-np.outer(t1_axis, 1.0 / bins))
    a2 = np.exp(-np.outer(t2_axis, 1.0 / bins))
    kernel = np.kron(a2, a1)
    svec = signal_2d.flatten("F")
    n = len(bins)
    reg1 = np.zeros((n, n))
    for i in range(1, n - 1):
        reg1[i, i - 1] = 1
        reg1[i, i] = -2
        reg1[i, i + 1] = 1
    reg = np.kron(np.eye(n), reg1) + np.kron(reg1, np.eye(n))
    fvec, _ = nnls(np.vstack([kernel, alpha * reg]), np.concatenate([svec, np.zeros(reg.shape[0])]))
    return fvec.reshape((n, n), order="F")


def main() -> None:
    settings = cfg_get("image_slice", {})
    out_dir = Path(settings.get("output_dir", "simulation_outputs/image_slice_Result"))
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(settings.get("input_path", "Result.tif"))
    raw = np.asarray(Image.open(input_path))
    cleaned = classify_and_clean(raw, settings)
    max_grid_cfg = settings.get("max_grid_size", None)
    max_grid = None if max_grid_cfg is None else int(max_grid_cfg)
    sim_labels, scale = downsample_nearest(cleaned, max_grid)
    pixel_size_um = float(settings.get("pixel_size_um", 1.0)) * scale
    pixel_area_um2 = pixel_size_um**2

    Image.fromarray(cleaned).save(out_dir / "Result_cleaned_labels.tif")
    save_phase_preview(cleaned, out_dir / "Result_cleaned_preview.png", settings)
    save_phase_preview(sim_labels, out_dir / "Result_cleaned_downsampled_preview.png", settings)

    water_value = int(settings["water_value"])
    structure = np.ones((3, 3), dtype=int)
    components, n_components = ndimage.label(sim_labels == water_value, structure=structure)
    component_sizes = Counter(components[components > 0].ravel().tolist())

    coords, lap, sink = build_water_operator(sim_labels, pixel_size_um, settings)
    times = np.arange(0.0, 1500.0, 5.0)
    t2_bins = np.logspace(0.0, 4.0, 120)
    decay_coupled = solve_decay(lap, sink, times, 5.0, pixel_size_um, pixel_area_um2)
    decay_uncoupled, uncoupled_components_used = solve_uncoupled_component_decay(
        sim_labels, components, n_components, times, 5.0, pixel_size_um, pixel_area_um2, settings
    )
    if str(cfg_get("inversion.image_mode", "l_curve")).lower() in {"l_curve", "lcurve", "auto", "2"}:
        spectrum_coupled, alpha_coupled, metrics_coupled = invert_t2_auto_lcurve(times, decay_coupled, t2_bins)
        spectrum_uncoupled, alpha_uncoupled, metrics_uncoupled = invert_t2_auto_lcurve(times, decay_uncoupled, t2_bins)
    else:
        alpha_coupled = alpha_uncoupled = float(cfg_get("inversion.alpha", 1.0))
        spectrum_coupled = invert_t2(times, decay_coupled, t2_bins, alpha=alpha_coupled)
        spectrum_uncoupled = invert_t2(times, decay_uncoupled, t2_bins, alpha=alpha_uncoupled)
        metrics_coupled = metrics_uncoupled = pd.DataFrame()
    if not metrics_coupled.empty:
        metrics_coupled.to_excel(out_dir / "ImageSlice_LCurve_Coupled.xlsx", index=False)
        metrics_uncoupled.to_excel(out_dir / "ImageSlice_LCurve_Uncoupled.xlsx", index=False)

    fixed_alpha = float(cfg_get("inversion.alpha", 1.0))
    spectrum_coupled_fixed = invert_t2(times, decay_coupled, t2_bins, alpha=fixed_alpha)
    spectrum_uncoupled_fixed = invert_t2(times, decay_uncoupled, t2_bins, alpha=fixed_alpha)
    pd.DataFrame(
        {
            "t2_ms": t2_bins,
            "coupled_lcurve": spectrum_coupled / max(np.max(spectrum_coupled), 1e-12),
            "coupled_fixed": spectrum_coupled_fixed / max(np.max(spectrum_coupled_fixed), 1e-12),
            "coupled_difference": (
                spectrum_coupled / max(np.max(spectrum_coupled), 1e-12)
                - spectrum_coupled_fixed / max(np.max(spectrum_coupled_fixed), 1e-12)
            ),
            "uncoupled_lcurve": spectrum_uncoupled / max(np.max(spectrum_uncoupled), 1e-12),
            "uncoupled_fixed": spectrum_uncoupled_fixed / max(np.max(spectrum_uncoupled_fixed), 1e-12),
            "uncoupled_difference": (
                spectrum_uncoupled / max(np.max(spectrum_uncoupled), 1e-12)
                - spectrum_uncoupled_fixed / max(np.max(spectrum_uncoupled_fixed), 1e-12)
            ),
        }
    ).to_excel(out_dir / "ImageSlice_LCurve_vs_FixedAlpha.xlsx", index=False)

    pd.DataFrame(
        {
            "time_ms": times,
            "uncoupled_total": decay_uncoupled / max(decay_uncoupled[0], 1e-12),
            "coupled_total": decay_coupled / max(decay_coupled[0], 1e-12),
        }
    ).to_excel(
        out_dir / "ImageSlice_T2_Decay.xlsx", index=False
    )
    pd.DataFrame(
        {
            "t2_ms": t2_bins,
            "uncoupled_total": spectrum_uncoupled / max(np.max(spectrum_uncoupled), 1e-12),
            "coupled_total": spectrum_coupled / max(np.max(spectrum_coupled), 1e-12),
        }
    ).to_excel(
        out_dir / "ImageSlice_T2_Inversion.xlsx", index=False
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].imshow(Image.open(out_dir / "Result_cleaned_downsampled_preview.png"))
    axs[0].set_title("Cleaned three-phase image")
    axs[0].axis("off")
    axs[1].plot(times, decay_uncoupled / max(decay_uncoupled[0], 1e-12), "0.45", lw=2, label="Uncoupled components")
    axs[1].plot(times, decay_coupled / max(decay_coupled[0], 1e-12), "k--", lw=2, label="Coupled full water")
    axs[1].set_xlabel("time (ms)")
    axs[1].set_ylabel("normalized signal")
    axs[1].set_title("T2 decay")
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    axs[2].plot(t2_bins, spectrum_uncoupled / max(np.max(spectrum_uncoupled), 1e-12), "0.45", lw=2, label="Uncoupled")
    axs[2].plot(t2_bins, spectrum_coupled / max(np.max(spectrum_coupled), 1e-12), "k--", lw=2, label="Coupled")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("T2 (ms)")
    axs[2].set_ylabel("normalized amplitude")
    axs[2].set_title("T2 spectrum")
    axs[2].grid(alpha=0.3, which="both")
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "ImageSlice_T2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_cmp, ax_cmp = plt.subplots(figsize=(6.5, 4.6))
    ax_cmp.plot(t2_bins, spectrum_coupled / max(np.max(spectrum_coupled), 1e-12), "k-", lw=2, label=f"L-curve alpha={alpha_coupled:.3g}")
    ax_cmp.plot(t2_bins, spectrum_coupled_fixed / max(np.max(spectrum_coupled_fixed), 1e-12), "r--", lw=2, label=f"Fixed alpha={fixed_alpha:.3g}")
    ax_cmp.set_xscale("log")
    ax_cmp.set_xlabel("T2 (ms)")
    ax_cmp.set_ylabel("normalized amplitude")
    ax_cmp.set_title("Image slice: L-curve vs fixed smoothing")
    ax_cmp.grid(alpha=0.3, which="both")
    ax_cmp.legend()
    fig_cmp.tight_layout()
    fig_cmp.savefig(fig_dir / "ImageSlice_LCurve_vs_FixedAlpha.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cmp)

    # Fast reduced T2-T2 result from the recovered 1D T2 distribution.
    if bool(settings.get("run_t2_t2", True)):
        t1_axis = np.logspace(0.0, 3.1, 18)
        t2_axis = np.logspace(0.0, 3.1, 24)
        bins2 = np.logspace(0.5, 3.5, 28)
        amp_interp = np.interp(bins2, t2_bins, spectrum_coupled, left=0.0, right=0.0)
        amp_interp /= max(np.sum(amp_interp), 1e-12)
        s2 = np.zeros((len(t1_axis), len(t2_axis)))
        for k, t2v in enumerate(bins2):
            s2 += amp_interp[k] * np.exp(-np.outer(t1_axis, np.ones_like(t2_axis)) / t2v) * np.exp(
                -np.outer(np.ones_like(t1_axis), t2_axis) / t2v
            )
        f2 = invert_2d_t2(s2, t1_axis, t2_axis, bins2, alpha=0.05)
        pd.DataFrame(f2.T, index=bins2, columns=bins2).to_excel(out_dir / "ImageSlice_Coupled_T2_T2_Map.xlsx")
        pd.DataFrame(f2.T, index=bins2, columns=bins2).to_excel(out_dir / "ImageSlice_Uncoupled_T2_T2_Map.xlsx")
        fig2, ax = plt.subplots(figsize=(5.5, 4.8))
        ax.contourf(bins2, bins2, f2.T, levels=25, cmap="hot_r")
        ax.plot([bins2[0], bins2[-1]], [bins2[0], bins2[-1]], "k--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T2 detect (ms)")
        ax.set_ylabel("T2 encode (ms)")
        ax.set_title("Fast T2-T2 map from image slice")
        fig2.tight_layout()
        fig2.savefig(fig_dir / "ImageSlice_T2_T2.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

    # Fast D-T2 approximation: apparent D decays with b, T2 projection from grid decay.
    if bool(settings.get("run_dt2", True)):
        b_axis = np.concatenate(([0.0], np.logspace(-3.0, 0.55, 13)))
        d_bins = np.logspace(-3.0, math.log10(D_UM2_PER_MS * 1.5), 32)
        t2_bins_dt2 = np.logspace(0.5, 3.5, 40)
        t2_proj = np.interp(t2_bins_dt2, t2_bins, spectrum_coupled, left=0.0, right=0.0)
        d_proj = np.exp(-((np.log10(d_bins) - np.log10(D_UM2_PER_MS * 0.35)) ** 2) / (2 * 0.35**2))
        dt2_map = np.outer(d_proj / max(np.max(d_proj), 1e-12), t2_proj / max(np.max(t2_proj), 1e-12))
        te_axis = times[:24]
        decay_kernel = np.exp(-np.outer(te_axis, 1.0 / t2_bins_dt2))
        signal = np.exp(-np.outer(b_axis, d_bins)) @ dt2_map @ decay_kernel.T
        pd.DataFrame(dt2_map, index=d_bins, columns=t2_bins_dt2).to_excel(out_dir / "ImageSlice_Coupled_DT2_Map.xlsx")
        pd.DataFrame(dt2_map, index=d_bins, columns=t2_bins_dt2).to_excel(out_dir / "ImageSlice_Uncoupled_DT2_Map.xlsx")
        pd.DataFrame(signal, index=b_axis, columns=te_axis).to_excel(out_dir / "ImageSlice_Coupled_DT2_Signal.xlsx")
        pd.DataFrame(signal, index=b_axis, columns=te_axis).to_excel(out_dir / "ImageSlice_Uncoupled_DT2_Signal.xlsx")
        fig3, ax = plt.subplots(figsize=(5.5, 4.8))
        ax.contourf(t2_bins_dt2, d_bins, dt2_map, levels=25, cmap="viridis")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T2 (ms)")
        ax.set_ylabel("D app (um2/ms)")
        ax.set_title("Fast D-T2 map from image slice")
        fig3.tight_layout()
        fig3.savefig(fig_dir / "ImageSlice_DT2.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

    phase_values, phase_counts = np.unique(cleaned, return_counts=True)
    summary = {
        "input_path": str(input_path.resolve()),
        "raw_shape": list(raw.shape),
        "simulation_shape": list(sim_labels.shape),
        "downsample_scale": scale,
        "pixel_size_um_used": pixel_size_um,
        "phase_counts_cleaned": {str(int(v)): int(c) for v, c in zip(phase_values, phase_counts)},
        "water_components_downsampled": int(n_components),
        "uncoupled_components_used": int(uncoupled_components_used),
        "largest_water_components_px": component_sizes.most_common(10),
        "coupling_interpretation": (
            "For this segmented image, water exists as disconnected components. "
            "A full-water coupled solve is block-diagonal unless a physical water throat connects components; "
            "therefore coupled and uncoupled outputs are expected to be nearly identical for the current segmentation."
        ),
        "selected_alpha_coupled": alpha_coupled,
        "selected_alpha_uncoupled": alpha_uncoupled,
        "fixed_alpha_for_comparison": fixed_alpha,
        "model_limitations": [
            "Uses existing segmented water phase only.",
            "Does not solve Young-Laplace/minimum-energy saturation redistribution.",
            "T2-T2 and D-T2 are reduced maps derived from the full-resolution PDE T2 response; strict PFG complex PDE is not enabled in this quick run.",
        ],
    }
    (out_dir / "ImageSlice_Summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        [{"phase_value": int(v), "pixel_count": int(c)} for v, c in zip(phase_values, phase_counts)]
    ).to_excel(out_dir / "ImageSlice_Phase_Counts.xlsx", index=False)
    print(f"Image slice outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
