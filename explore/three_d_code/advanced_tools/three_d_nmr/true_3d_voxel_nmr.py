"""True 3D voxel-based Bloch-Torrey T2 simulation for segmented CT stacks.

This script accepts a 3D binary digital rock directly. It first classifies
open and closed pore components in 3D, then solves a 3D finite-volume/finite-
difference T2 decay problem only on open, NMR-responsive pore voxels.

The discretized equation is:

    du/dt = D * Laplacian(u) - u/T2B - rho * (S/V) * u

where the Laplacian is assembled from 6-neighbor open pore voxel faces. Faces
against solid or 3D-closed pores receive solid surface-relaxation. External
volume faces are no-flux by default and can be changed explicitly with
``--external-boundary relax``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, factorized

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from advanced_tools.batch_tiff_random_slices_nmr import save_inversion_outputs
from advanced_tools.three_d_nmr.open_pore_workflow import (
    DEFAULT_TIFF,
    OpenPoreDetectionResult,
    component_table,
    detect_open_pores,
    save_detection_summary,
)
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


@dataclass(frozen=True)
class VoxelNmrParams:
    voxel_size_um: float = 1.70
    diffusion_um2_per_ms: float = 2.0
    bulk_t2_ms: float = 3000.0
    rho_solid_um_per_ms: float = 0.005
    rho_external_um_per_ms: float = 0.0
    dt_ms: float = 5.0
    t_max_ms: float = 1500.0
    external_boundary: str = "no_flux"


@dataclass
class VoxelDecaySystem:
    lhs: csr_matrix
    open_voxels: int
    internal_neighbor_pairs: int
    solid_boundary_faces: int
    external_boundary_faces: int
    responsive_boundary_faces: int
    pore_volume_um3: float
    matrix_nonzeros: int
    voxel_size_um: float


@dataclass
class VoxelMatrixFreeSystem:
    diagonal: np.ndarray
    neighbor_pairs: list[tuple[np.ndarray, np.ndarray]]
    conductance: float
    open_voxels: int
    internal_neighbor_pairs: int
    solid_boundary_faces: int
    external_boundary_faces: int
    responsive_boundary_faces: int
    pore_volume_um3: float
    matrix_nonzeros: int
    voxel_size_um: float


def downsample_binary_volume(
    volume: np.ndarray,
    *,
    factor: int,
    pore_value: int,
    solid_value: int,
    pore_fraction_threshold: float,
) -> np.ndarray:
    if factor <= 1:
        return volume.copy()
    if not (0.0 < pore_fraction_threshold <= 1.0):
        raise ValueError("pore_fraction_threshold must satisfy 0 < threshold <= 1.")

    z_trim = (volume.shape[0] // factor) * factor
    y_trim = (volume.shape[1] // factor) * factor
    x_trim = (volume.shape[2] // factor) * factor
    if min(z_trim, y_trim, x_trim) <= 0:
        raise ValueError(f"Downsample factor {factor} is too large for volume shape {volume.shape}.")

    pore = volume[:z_trim, :y_trim, :x_trim] == pore_value
    block = pore.reshape(
        z_trim // factor,
        factor,
        y_trim // factor,
        factor,
        x_trim // factor,
        factor,
    )
    pore_fraction = block.mean(axis=(1, 3, 5))
    out = np.full(pore_fraction.shape, solid_value, dtype=volume.dtype)
    out[pore_fraction >= pore_fraction_threshold] = pore_value
    return out


def _add_at(indices: np.ndarray, values: np.ndarray, amount: int = 1) -> None:
    if indices.size:
        np.add.at(values, indices.astype(np.int64, copy=False), amount)


def build_3d_decay_system(open_mask: np.ndarray, params: VoxelNmrParams) -> VoxelDecaySystem:
    if open_mask.ndim != 3:
        raise ValueError(f"open_mask must be 3D, got shape {open_mask.shape}.")
    if params.voxel_size_um <= 0:
        raise ValueError("voxel_size_um must be positive.")
    if params.external_boundary not in {"no_flux", "relax"}:
        raise ValueError("external_boundary must be 'no_flux' or 'relax'.")

    open_voxels = int(np.sum(open_mask))
    if open_voxels == 0:
        raise ValueError("No open pore voxels are available for 3D NMR simulation.")

    voxel_index = np.full(open_mask.shape, -1, dtype=np.int64)
    voxel_index[open_mask] = np.arange(open_voxels, dtype=np.int64)
    neighbor_degree = np.zeros(open_voxels, dtype=np.int16)
    solid_boundary_count = np.zeros(open_voxels, dtype=np.int16)
    external_boundary_count = np.zeros(open_voxels, dtype=np.int16)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    internal_pairs = 0

    axis_slices = [
        ((slice(0, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
        ((slice(None), slice(0, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        ((slice(None), slice(None), slice(0, -1)), (slice(None), slice(None), slice(1, None))),
    ]
    conductance = params.diffusion_um2_per_ms * params.dt_ms / (params.voxel_size_um**2)

    for left_slice, right_slice in axis_slices:
        left_open = open_mask[left_slice]
        right_open = open_mask[right_slice]
        both_open = left_open & right_open
        left_ids = voxel_index[left_slice]
        right_ids = voxel_index[right_slice]

        if np.any(both_open):
            a = left_ids[both_open]
            b = right_ids[both_open]
            internal_pairs += int(a.size)
            _add_at(a, neighbor_degree)
            _add_at(b, neighbor_degree)
            rows.extend([a, b])
            cols.extend([b, a])
            data.extend(
                [
                    np.full(a.size, -conductance, dtype=float),
                    np.full(b.size, -conductance, dtype=float),
                ]
            )

        left_boundary = left_open & ~right_open
        right_boundary = right_open & ~left_open
        _add_at(left_ids[left_boundary], solid_boundary_count)
        _add_at(right_ids[right_boundary], solid_boundary_count)

    boundary_faces = [
        open_mask[0, :, :],
        open_mask[-1, :, :],
        open_mask[:, 0, :],
        open_mask[:, -1, :],
        open_mask[:, :, 0],
        open_mask[:, :, -1],
    ]
    boundary_ids = [
        voxel_index[0, :, :],
        voxel_index[-1, :, :],
        voxel_index[:, 0, :],
        voxel_index[:, -1, :],
        voxel_index[:, :, 0],
        voxel_index[:, :, -1],
    ]
    for face_open, face_ids in zip(boundary_faces, boundary_ids):
        _add_at(face_ids[face_open], external_boundary_count)

    responsive_boundary_count = solid_boundary_count.astype(float)
    if params.external_boundary == "relax":
        responsive_boundary_count = responsive_boundary_count + external_boundary_count.astype(float)

    sink = (
        params.dt_ms / params.bulk_t2_ms
        + params.rho_solid_um_per_ms * params.dt_ms * solid_boundary_count.astype(float) / params.voxel_size_um
    )
    if params.external_boundary == "relax":
        sink += (
            params.rho_external_um_per_ms
            * params.dt_ms
            * external_boundary_count.astype(float)
            / params.voxel_size_um
        )

    diagonal = 1.0 + conductance * neighbor_degree.astype(float) + sink
    row_arr = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    col_arr = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    data_arr = np.concatenate(data) if data else np.array([], dtype=float)
    lhs = diags(diagonal, format="csr") + csr_matrix((data_arr, (row_arr, col_arr)), shape=(open_voxels, open_voxels))
    lhs = lhs.tocsr()

    return VoxelDecaySystem(
        lhs=lhs,
        open_voxels=open_voxels,
        internal_neighbor_pairs=int(internal_pairs),
        solid_boundary_faces=int(np.sum(solid_boundary_count)),
        external_boundary_faces=int(np.sum(external_boundary_count)),
        responsive_boundary_faces=int(np.sum(responsive_boundary_count)),
        pore_volume_um3=float(open_voxels * params.voxel_size_um**3),
        matrix_nonzeros=int(lhs.nnz),
        voxel_size_um=float(params.voxel_size_um),
    )


def build_3d_decay_system_matrix_free(open_mask: np.ndarray, params: VoxelNmrParams) -> VoxelMatrixFreeSystem:
    if open_mask.ndim != 3:
        raise ValueError(f"open_mask must be 3D, got shape {open_mask.shape}.")
    if params.voxel_size_um <= 0:
        raise ValueError("voxel_size_um must be positive.")
    if params.external_boundary not in {"no_flux", "relax"}:
        raise ValueError("external_boundary must be 'no_flux' or 'relax'.")

    open_voxels = int(np.sum(open_mask))
    if open_voxels == 0:
        raise ValueError("No open pore voxels are available for 3D NMR simulation.")

    if open_voxels > np.iinfo(np.int32).max:
        index_dtype = np.int64
    else:
        index_dtype = np.int32
    voxel_index = np.full(open_mask.shape, -1, dtype=index_dtype)
    voxel_index[open_mask] = np.arange(open_voxels, dtype=index_dtype)
    neighbor_degree = np.zeros(open_voxels, dtype=np.int16)
    solid_boundary_count = np.zeros(open_voxels, dtype=np.int16)
    external_boundary_count = np.zeros(open_voxels, dtype=np.int16)
    neighbor_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    internal_pairs = 0

    axis_slices = [
        ((slice(0, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
        ((slice(None), slice(0, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        ((slice(None), slice(None), slice(0, -1)), (slice(None), slice(None), slice(1, None))),
    ]
    conductance = params.diffusion_um2_per_ms * params.dt_ms / (params.voxel_size_um**2)

    for left_slice, right_slice in axis_slices:
        left_open = open_mask[left_slice]
        right_open = open_mask[right_slice]
        both_open = left_open & right_open
        left_ids = voxel_index[left_slice]
        right_ids = voxel_index[right_slice]

        if np.any(both_open):
            a = left_ids[both_open].astype(index_dtype, copy=False)
            b = right_ids[both_open].astype(index_dtype, copy=False)
            internal_pairs += int(a.size)
            _add_at(a, neighbor_degree)
            _add_at(b, neighbor_degree)
            neighbor_pairs.append((a.copy(), b.copy()))

        left_boundary = left_open & ~right_open
        right_boundary = right_open & ~left_open
        _add_at(left_ids[left_boundary], solid_boundary_count)
        _add_at(right_ids[right_boundary], solid_boundary_count)

    boundary_faces = [
        open_mask[0, :, :],
        open_mask[-1, :, :],
        open_mask[:, 0, :],
        open_mask[:, -1, :],
        open_mask[:, :, 0],
        open_mask[:, :, -1],
    ]
    boundary_ids = [
        voxel_index[0, :, :],
        voxel_index[-1, :, :],
        voxel_index[:, 0, :],
        voxel_index[:, -1, :],
        voxel_index[:, :, 0],
        voxel_index[:, :, -1],
    ]
    for face_open, face_ids in zip(boundary_faces, boundary_ids):
        _add_at(face_ids[face_open], external_boundary_count)

    responsive_boundary_count = solid_boundary_count.astype(float)
    if params.external_boundary == "relax":
        responsive_boundary_count = responsive_boundary_count + external_boundary_count.astype(float)

    sink = (
        params.dt_ms / params.bulk_t2_ms
        + params.rho_solid_um_per_ms * params.dt_ms * solid_boundary_count.astype(float) / params.voxel_size_um
    )
    if params.external_boundary == "relax":
        sink += (
            params.rho_external_um_per_ms
            * params.dt_ms
            * external_boundary_count.astype(float)
            / params.voxel_size_um
        )

    diagonal = 1.0 + conductance * neighbor_degree.astype(float) + sink
    return VoxelMatrixFreeSystem(
        diagonal=diagonal.astype(np.float64, copy=False),
        neighbor_pairs=neighbor_pairs,
        conductance=float(conductance),
        open_voxels=open_voxels,
        internal_neighbor_pairs=int(internal_pairs),
        solid_boundary_faces=int(np.sum(solid_boundary_count)),
        external_boundary_faces=int(np.sum(external_boundary_count)),
        responsive_boundary_faces=int(np.sum(responsive_boundary_count)),
        pore_volume_um3=float(open_voxels * params.voxel_size_um**3),
        matrix_nonzeros=int(open_voxels + 2 * internal_pairs),
        voxel_size_um=float(params.voxel_size_um),
    )


def matrix_free_linear_operator(system: VoxelMatrixFreeSystem) -> LinearOperator:
    n = int(system.open_voxels)
    conductance = float(system.conductance)

    def matvec(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y = system.diagonal * x_arr
        for left, right in system.neighbor_pairs:
            y[left] -= conductance * x_arr[right]
            y[right] -= conductance * x_arr[left]
        return y

    return LinearOperator((n, n), matvec=matvec, dtype=np.float64)


def matrix_free_jacobi_preconditioner(system: VoxelMatrixFreeSystem) -> LinearOperator:
    n = int(system.open_voxels)
    inverse_diagonal = 1.0 / system.diagonal
    return LinearOperator((n, n), matvec=lambda x: inverse_diagonal * x, dtype=np.float64)


def solve_3d_decay(
    system: VoxelDecaySystem,
    params: VoxelNmrParams,
    *,
    linear_solver: str = "factorized",
    cg_rtol: float = 1e-8,
    cg_maxiter: int = 1000,
    progress_interval: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, params.t_max_ms + 0.5 * params.dt_ms, params.dt_ms)
    magnetization = np.ones(system.open_voxels, dtype=float)
    signal = np.empty(times.size, dtype=float)
    voxel_volume = params.voxel_size_um**3

    if linear_solver == "factorized":
        solve = factorized(system.lhs.tocsc())
        for i in range(times.size):
            signal[i] = float(np.sum(magnetization) * voxel_volume)
            if i == times.size - 1:
                break
            magnetization = solve(magnetization)
    elif linear_solver == "cg":
        for i in range(times.size):
            signal[i] = float(np.sum(magnetization) * voxel_volume)
            if i == times.size - 1:
                break
            next_magnetization, info = cg(
                system.lhs,
                magnetization,
                x0=magnetization,
                rtol=cg_rtol,
                atol=0.0,
                maxiter=cg_maxiter,
            )
            if info != 0:
                raise RuntimeError(f"CG solve failed at time index {i} with info={info}.")
            magnetization = next_magnetization
            if progress_interval > 0 and ((i + 1) % progress_interval == 0):
                print(f"  solved time step {i + 1}/{times.size - 1} ({times[i + 1]:g} ms)", flush=True)
    else:
        raise ValueError("linear_solver must be 'factorized' or 'cg'.")

    return times, signal


def solve_3d_decay_matrix_free(
    system: VoxelMatrixFreeSystem,
    params: VoxelNmrParams,
    *,
    cg_rtol: float = 1e-6,
    cg_maxiter: int = 300,
    progress_interval: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    times = np.arange(0.0, params.t_max_ms + 0.5 * params.dt_ms, params.dt_ms)
    magnetization = np.ones(system.open_voxels, dtype=np.float64)
    signal = np.empty(times.size, dtype=np.float64)
    voxel_volume = params.voxel_size_um**3
    operator = matrix_free_linear_operator(system)
    preconditioner = matrix_free_jacobi_preconditioner(system)
    rows = []

    for i in range(times.size):
        signal[i] = float(np.sum(magnetization) * voxel_volume)
        if i == times.size - 1:
            rows.append({"time_index": i, "time_ms": float(times[i]), "cg_info": 0, "cg_iterations": 0})
            break
        iterations = 0

        def count_iteration(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        next_magnetization, info = cg(
            operator,
            magnetization,
            x0=magnetization,
            rtol=cg_rtol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=preconditioner,
            callback=count_iteration,
        )
        rows.append({"time_index": i, "time_ms": float(times[i]), "cg_info": int(info), "cg_iterations": int(iterations)})
        if info != 0:
            raise RuntimeError(f"Matrix-free CG solve failed at time index {i} with info={info}.")
        magnetization = next_magnetization
        if progress_interval > 0 and ((i + 1) % progress_interval == 0):
            print(
                f"  solved time step {i + 1}/{times.size - 1} "
                f"({times[i + 1]:g} ms, cg_iterations={iterations})",
                flush=True,
            )

    return times, signal, pd.DataFrame(rows)


def save_decay_plot(time_ms: np.ndarray, normalized_signal: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(time_ms, normalized_signal, color="black", lw=2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized NMR signal")
    ax.set_title("True 3D voxel T2 decay")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_projection_preview(open_mask: np.ndarray, output_path: Path) -> None:
    projections = [
        np.max(open_mask, axis=0),
        np.max(open_mask, axis=1),
        np.max(open_mask, axis=2),
    ]
    titles = ["max projection Z", "max projection Y", "max projection X"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, projection, title in zip(axs, projections, titles):
        ax.imshow(projection, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_89_true_3d_voxel_nmr"))
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    parser.add_argument("--voxel-size-um", type=float, default=1.70)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--downsample-pore-threshold", type=float, default=0.5)
    parser.add_argument("--max-open-voxels", type=int, default=2_000_000)
    parser.add_argument("--diffusion-um2-per-ms", type=float, default=2.0)
    parser.add_argument("--bulk-t2-ms", type=float, default=3000.0)
    parser.add_argument("--rho-solid-um-per-ms", type=float, default=0.005)
    parser.add_argument("--rho-external-um-per-ms", type=float, default=0.0)
    parser.add_argument("--external-boundary", choices=["no_flux", "relax"], default="no_flux")
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--t-max-ms", type=float, default=1500.0)
    parser.add_argument("--assembly", choices=["csr", "matrix_free"], default="matrix_free")
    parser.add_argument("--linear-solver", choices=["factorized", "cg"], default="cg")
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    parser.add_argument("--t2-bins", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not args.input_tiff.exists():
        raise FileNotFoundError(f"Input TIFF not found: {args.input_tiff}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 3D TIFF: {args.input_tiff}")
    raw_volume = tifffile.imread(str(args.input_tiff))
    if raw_volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF stack, got shape {raw_volume.shape}.")

    effective_voxel_size_um = float(args.voxel_size_um) * int(args.downsample_factor)
    print(f"Downsampling factor: {args.downsample_factor}; effective voxel size: {effective_voxel_size_um:g} um")
    volume = downsample_binary_volume(
        raw_volume,
        factor=int(args.downsample_factor),
        pore_value=int(args.pore_value),
        solid_value=int(args.solid_value),
        pore_fraction_threshold=float(args.downsample_pore_threshold),
    )

    print("Detecting 3D open pore components on simulation volume...")
    detection: OpenPoreDetectionResult = detect_open_pores(
        volume,
        pore_value=int(args.pore_value),
        solid_value=int(args.solid_value),
        connectivity=int(args.connectivity),
    )
    if detection.open_pore_voxels > int(args.max_open_voxels):
        raise RuntimeError(
            f"Open pore voxels ({detection.open_pore_voxels}) exceed --max-open-voxels={args.max_open_voxels}. "
            "Increase --downsample-factor or raise the guard after checking memory/runtime."
        )

    detection_summary = save_detection_summary(
        detection,
        tuple(int(v) for v in volume.shape),
        effective_voxel_size_um,
        args.output_dir / "open_closed_pore_summary.csv",
    )
    component_table(detection, effective_voxel_size_um).to_csv(
        args.output_dir / "pore_component_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_projection_preview(detection.open_mask, args.output_dir / "open_pore_3d_projection_preview.png")

    params = VoxelNmrParams(
        voxel_size_um=effective_voxel_size_um,
        diffusion_um2_per_ms=float(args.diffusion_um2_per_ms),
        bulk_t2_ms=float(args.bulk_t2_ms),
        rho_solid_um_per_ms=float(args.rho_solid_um_per_ms),
        rho_external_um_per_ms=float(args.rho_external_um_per_ms),
        dt_ms=float(args.dt_ms),
        t_max_ms=float(args.t_max_ms),
        external_boundary=str(args.external_boundary),
    )
    print("Assembling true 3D voxel PDE system...")
    if args.assembly == "matrix_free":
        system = build_3d_decay_system_matrix_free(detection.open_mask, params)
        operator_summary = {
            "assembly": "matrix_free",
            "open_voxels": system.open_voxels,
            "internal_neighbor_pairs": system.internal_neighbor_pairs,
            "solid_boundary_faces": system.solid_boundary_faces,
            "external_boundary_faces": system.external_boundary_faces,
            "responsive_boundary_faces": system.responsive_boundary_faces,
            "pore_volume_um3": system.pore_volume_um3,
            "matrix_nonzeros_equivalent": system.matrix_nonzeros,
            "voxel_size_um": system.voxel_size_um,
            "stored_neighbor_pair_blocks": len(system.neighbor_pairs),
        }
    else:
        system = build_3d_decay_system(detection.open_mask, params)
        operator_summary = {
            "assembly": "csr",
            **asdict(system),
            "lhs": None,
        }
        operator_summary.pop("lhs", None)
    operator_summary.update({
        "boundary_condition_internal_nonopen_faces": (
            "solid/closed-pore faces use rho_solid surface relaxation"
        ),
        "boundary_condition_external_faces": args.external_boundary,
    })
    pd.DataFrame([operator_summary]).to_csv(args.output_dir / "operator_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Solving 3D T2 decay for {system.open_voxels} open voxels...")
    if args.assembly == "matrix_free":
        time_ms, signal, cg_history = solve_3d_decay_matrix_free(
            system,
            params,
            cg_rtol=float(args.cg_rtol),
            cg_maxiter=int(args.cg_maxiter),
            progress_interval=int(args.progress_interval),
        )
        cg_history.to_csv(args.output_dir / "cg_iteration_history.csv", index=False)
        solver_outputs = {"cg_iteration_history_csv": str((args.output_dir / "cg_iteration_history.csv").resolve())}
    else:
        time_ms, signal = solve_3d_decay(
            system,
            params,
            linear_solver=str(args.linear_solver),
            cg_rtol=float(args.cg_rtol),
            cg_maxiter=int(args.cg_maxiter),
            progress_interval=int(args.progress_interval),
        )
        solver_outputs = {}
    normalized = signal / max(float(signal[0]), 1e-30)
    decay_path = args.output_dir / "true3d_nmr_decay.csv"
    pd.DataFrame({"time_ms": time_ms, "signal": signal, "normalized_signal": normalized}).to_csv(decay_path, index=False)
    save_decay_plot(time_ms, normalized, args.output_dir / "true3d_nmr_decay.png")

    inversion_cfg = NnlsConfig(
        num_bins=int(args.t2_bins),
        regularization=float(args.fixed_alpha),
        t2_min_ms=float(args.t2_min_ms),
        t2_max_ms=float(args.t2_max_ms),
    )
    inversion = invert_single_signal_nnls(time_ms, normalized, signal_name="true3d_voxel_nmr", config=inversion_cfg)
    inversion_outputs = save_inversion_outputs(inversion, args.output_dir / "true3d_voxel_nmr")

    manifest = {
        "input_tiff": str(args.input_tiff.resolve()),
        "source_label_convention": {
            str(args.pore_value): "pore voxel before open/closed filtering",
            str(args.solid_value): "solid matrix",
        },
        "raw_volume_shape": list(raw_volume.shape),
        "simulation_volume_shape": list(volume.shape),
        "downsample_factor": int(args.downsample_factor),
        "downsample_pore_threshold": float(args.downsample_pore_threshold),
        "raw_voxel_size_um": float(args.voxel_size_um),
        "effective_voxel_size_um": effective_voxel_size_um,
        "open_pore_definition": (
            "A pore component is open/NMR-responsive only if it is connected in 3D to at least one external "
            "volume face after any explicit downsampling."
        ),
        "model_dimension_note": (
            "This is a true 3D voxel-grid PDE solve on the open pore domain. It does not sample 2D slices."
        ),
        "connectivity": int(args.connectivity),
        "nmr_params": asdict(params),
        "assembly": str(args.assembly),
        "linear_solver": "cg" if args.assembly == "matrix_free" else str(args.linear_solver),
        "inversion_config": inversion_cfg.__dict__,
        "detection_summary": detection_summary,
        "operator_summary": operator_summary,
        "outputs": {
            "decay_csv": str(decay_path.resolve()),
            "decay_png": str((args.output_dir / "true3d_nmr_decay.png").resolve()),
            "inversion_csv": inversion_outputs["spectrum_csv"],
            "inversion_png": inversion_outputs["figure_png"],
            "operator_summary_csv": str((args.output_dir / "operator_summary.csv").resolve()),
            "open_closed_pore_summary_csv": str((args.output_dir / "open_closed_pore_summary.csv").resolve()),
            "pore_component_table_csv": str((args.output_dir / "pore_component_table.csv").resolve()),
            "projection_preview_png": str((args.output_dir / "open_pore_3d_projection_preview.png").resolve()),
            **solver_outputs,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
