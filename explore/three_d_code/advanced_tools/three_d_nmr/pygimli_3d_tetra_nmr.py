"""3D tetrahedral NMR workflow for segmented binary CT volumes.

This workflow is intentionally scoped to small 3D subvolumes. It converts each
3D-open pore voxel into six conforming tetrahedra, stores the result as a
pyGIMLi ``Mesh``, and solves a P1 finite-element Bloch-Torrey T2 decay model on
the same tetrahedral node set.

The mesh is voxel-conformal: it preserves the segmented pore topology and
avoids requiring an external TetGen executable, which is not always installed
with pyGIMLi on Windows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg
from skimage import measure

try:
    import pygimli as pg
    import pygimli.meshtools as mt
except Exception:  # pragma: no cover - optional dependency
    pg = None
    mt = None

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
from advanced_tools.three_d_nmr.true_3d_voxel_nmr import save_projection_preview
from nmr_t2.config import NnlsConfig
from nmr_t2.nnls import invert_single_signal_nnls


DEFAULT_150_TIFF = Path(
    r"C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\sip模拟"
    r"\data_inventory\ct_backed_samples_raw_copy_20260605\sample_89_Grainstone"
    r"\CT_slices\89seged-150_150_150.tiff"
)

TET_PATTERN = np.asarray(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int32,
)

CORNER_OFFSETS_ZYX = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.int32,
)

FACE_TRIANGLES = {
    "z_minus": np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
    "z_plus": np.asarray([[4, 5, 7], [4, 7, 6]], dtype=np.int32),
    "y_minus": np.asarray([[0, 4, 5], [0, 5, 1]], dtype=np.int32),
    "y_plus": np.asarray([[2, 3, 7], [2, 7, 6]], dtype=np.int32),
    "x_minus": np.asarray([[0, 2, 6], [0, 6, 4]], dtype=np.int32),
    "x_plus": np.asarray([[1, 3, 7], [1, 7, 5]], dtype=np.int32),
}


@dataclass(frozen=True)
class TetraNmrParams:
    voxel_size_um: float = 1.70
    diffusion_um2_per_ms: float = 2.0
    bulk_t2_ms: float = 3000.0
    rho_solid_um_per_ms: float = 0.005
    rho_external_um_per_ms: float = 0.0
    dt_ms: float = 5.0
    t_max_ms: float = 1500.0
    external_boundary: str = "no_flux"


@dataclass
class VoxelTetraMesh:
    nodes_xyz_um: np.ndarray
    tets: np.ndarray
    voxel_corner_ids: np.ndarray
    open_voxel_coords_zyx: np.ndarray
    boundary_triangles_all: np.ndarray
    solid_boundary_triangles: np.ndarray
    external_boundary_triangles: np.ndarray
    solid_boundary_area_lumped: np.ndarray
    external_boundary_area_lumped: np.ndarray
    solid_boundary_faces: int
    external_boundary_faces: int


@dataclass
class TetraFemSystem:
    stiffness: coo_matrix
    mass_lumped: np.ndarray
    solid_boundary_lumped: np.ndarray
    external_boundary_lumped: np.ndarray
    node_count: int
    tetra_count: int
    stiffness_nonzeros: int


class NativeMeshingResourcePause(RuntimeError):
    def __init__(self, message: str, details: dict[str, object]):
        super().__init__(message)
        self.details = details


def tetgen_available(tetgen_executable: str = "tetgen") -> bool:
    return shutil.which(tetgen_executable) is not None


def assert_native_pygimli_mesher_available(tetgen_executable: str = "tetgen") -> None:
    if pg is None or mt is None:
        raise RuntimeError("pyGIMLi native 3D meshing requires pyGIMLi to be importable.")
    if not tetgen_available(tetgen_executable):
        raise RuntimeError(
            "pyGIMLi native 3D meshing requires TetGen on PATH. "
            f"Missing executable: {tetgen_executable!r}."
        )


def select_simulation_mask(
    volume: np.ndarray,
    detection: OpenPoreDetectionResult,
    *,
    pore_domain: str,
    pore_value: int,
) -> tuple[np.ndarray, int]:
    if pore_domain == "open":
        return detection.open_mask, int(detection.open_pore_voxels)
    if pore_domain == "all":
        mask = volume == int(pore_value)
        return mask, int(detection.total_pore_voxels)
    raise ValueError("pore_domain must be 'open' or 'all'.")


def downsample_binary_volume_preserve_pore_volume(
    volume: np.ndarray,
    *,
    pore_value: int,
    solid_value: int,
    factor: int,
    voxel_size_um: float,
) -> tuple[np.ndarray, dict[str, int | float | str | list[int]]]:
    factor = int(factor)
    if factor < 1:
        raise ValueError("downsample factor must be >= 1.")
    if factor == 1:
        stats = {
            "method": "none",
            "factor": 1,
            "original_shape_zyx": [int(v) for v in volume.shape],
            "padded_shape_zyx": [int(v) for v in volume.shape],
            "pad_width_zyx": [[0, 0], [0, 0], [0, 0]],
            "padding_label": int(solid_value),
            "coarse_shape_zyx": [int(v) for v in volume.shape],
            "original_voxel_size_um": float(voxel_size_um),
            "coarse_voxel_size_um": float(voxel_size_um),
            "original_pore_voxels": int(np.count_nonzero(volume == int(pore_value))),
            "target_coarse_pore_voxels": int(np.count_nonzero(volume == int(pore_value))),
            "represented_original_pore_voxels": int(np.count_nonzero(volume == int(pore_value))),
            "absolute_pore_voxel_error": 0,
            "porosity_absolute_error": 0.0,
        }
        return volume.copy(), stats

    pad_width = []
    for size in volume.shape:
        after = (factor - (int(size) % factor)) % factor
        pad_width.append((0, after))
    padded_volume = np.pad(volume, pad_width, mode="constant", constant_values=int(solid_value))
    padded_shape = tuple(int(v) for v in padded_volume.shape)
    pore_mask = padded_volume == int(pore_value)
    original_pore_voxels = int(np.count_nonzero(pore_mask))
    block_volume = factor**3
    coarse_shape = tuple(int(size) // factor for size in padded_shape)
    block_counts = pore_mask.reshape(
        coarse_shape[0],
        factor,
        coarse_shape[1],
        factor,
        coarse_shape[2],
        factor,
    ).sum(axis=(1, 3, 5))
    flat_counts = block_counts.ravel()
    target_coarse_pore_voxels = int(np.rint(original_pore_voxels / block_volume))
    target_coarse_pore_voxels = max(0, min(target_coarse_pore_voxels, flat_counts.size))
    order = np.lexsort((np.arange(flat_counts.size), -flat_counts))
    coarse_flat = np.full(flat_counts.size, int(solid_value), dtype=volume.dtype)
    if target_coarse_pore_voxels > 0:
        coarse_flat[order[:target_coarse_pore_voxels]] = int(pore_value)
    coarse = coarse_flat.reshape(coarse_shape)
    represented_original_pore_voxels = int(target_coarse_pore_voxels * block_volume)
    absolute_pore_voxel_error = int(abs(represented_original_pore_voxels - original_pore_voxels))
    total_voxels = int(volume.size)
    stats = {
        "method": "volume_preserving_binary",
        "factor": factor,
        "original_shape_zyx": [int(v) for v in volume.shape],
        "padded_shape_zyx": [int(v) for v in padded_shape],
        "pad_width_zyx": [[int(before), int(after)] for before, after in pad_width],
        "padding_label": int(solid_value),
        "coarse_shape_zyx": [int(v) for v in coarse_shape],
        "original_voxel_size_um": float(voxel_size_um),
        "coarse_voxel_size_um": float(voxel_size_um) * factor,
        "original_pore_voxels": original_pore_voxels,
        "target_coarse_pore_voxels": target_coarse_pore_voxels,
        "represented_original_pore_voxels": represented_original_pore_voxels,
        "absolute_pore_voxel_error": absolute_pore_voxel_error,
        "relative_pore_volume_error": float(
            absolute_pore_voxel_error / max(1, original_pore_voxels)
        ),
        "original_porosity_fraction": float(original_pore_voxels / total_voxels),
        "coarse_porosity_fraction": float(represented_original_pore_voxels / max(1, total_voxels)),
        "porosity_absolute_error": float(absolute_pore_voxel_error / max(1, total_voxels)),
        "tie_break": "descending pore occupancy, then increasing z-y-x block index",
    }
    return coarse, stats


def _neighbor_open(open_mask: np.ndarray, coords: np.ndarray, axis: int, delta: int) -> tuple[np.ndarray, np.ndarray]:
    neighbor = coords.copy()
    neighbor[:, axis] += delta
    inside = (neighbor[:, axis] >= 0) & (neighbor[:, axis] < open_mask.shape[axis])
    is_open = np.zeros(coords.shape[0], dtype=bool)
    if np.any(inside):
        valid = neighbor[inside]
        is_open[inside] = open_mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return inside, is_open


def _collect_boundary_triangles(
    open_mask: np.ndarray,
    coords: np.ndarray,
    voxel_corner_ids: np.ndarray,
    *,
    include_external: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    face_specs = [
        (0, -1, "z_minus"),
        (0, 1, "z_plus"),
        (1, -1, "y_minus"),
        (1, 1, "y_plus"),
        (2, -1, "x_minus"),
        (2, 1, "x_plus"),
    ]
    solid_triangles: list[np.ndarray] = []
    external_triangles: list[np.ndarray] = []
    solid_faces = 0
    external_faces = 0

    for axis, delta, face_name in face_specs:
        inside, is_open = _neighbor_open(open_mask, coords, axis, delta)
        solid_face = inside & ~is_open
        external_face = ~inside
        if np.any(solid_face):
            face_corners = FACE_TRIANGLES[face_name]
            tris = voxel_corner_ids[solid_face][:, face_corners].reshape(-1, 3)
            solid_triangles.append(tris.astype(np.int32, copy=False))
            solid_faces += int(np.count_nonzero(solid_face))
        if include_external and np.any(external_face):
            face_corners = FACE_TRIANGLES[face_name]
            tris = voxel_corner_ids[external_face][:, face_corners].reshape(-1, 3)
            external_triangles.append(tris.astype(np.int32, copy=False))
        external_faces += int(np.count_nonzero(external_face))

    solid = np.vstack(solid_triangles) if solid_triangles else np.empty((0, 3), dtype=np.int32)
    external = np.vstack(external_triangles) if external_triangles else np.empty((0, 3), dtype=np.int32)
    all_tris = np.vstack([solid, external]) if external.size else solid.copy()
    return all_tris, solid, external, solid_faces, external_faces


def _lump_triangle_area(nodes_xyz: np.ndarray, triangles: np.ndarray, node_count: int) -> np.ndarray:
    lumped = np.zeros(node_count, dtype=np.float64)
    if triangles.size == 0:
        return lumped
    pts = nodes_xyz[triangles]
    area = 0.5 * np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
    np.add.at(lumped, triangles.ravel(), np.repeat(area / 3.0, 3))
    return lumped


def _component_region_marker_positions(mask: np.ndarray, voxel_size_um: float, max_markers: int = 20000) -> np.ndarray:
    labels = measure.label(mask, connectivity=1)
    component_count = int(labels.max())
    if component_count > max_markers:
        message = (
            f"Pore component count ({component_count}) exceeds native pyGIMLi region-marker guard {max_markers}. "
            "This run is paused instead of coarsening or merging pore components."
        )
        raise NativeMeshingResourcePause(
            message,
            {
                "pause_stage": "region_marker_guard",
                "component_count": component_count,
                "max_region_markers": int(max_markers),
                "downsampling": "none",
                "geometry_simplification": "none",
            },
        )
    coords = np.argwhere(labels > 0)
    component_labels = labels[labels > 0].astype(np.int64, copy=False)
    counts = np.bincount(component_labels, minlength=component_count + 1).astype(float)
    sums = [
        np.bincount(component_labels, weights=coords[:, axis], minlength=component_count + 1)
        for axis in range(3)
    ]
    centers_zyx = np.column_stack([sums[axis][1:] / counts[1:] for axis in range(3)]) + 0.5
    return centers_zyx[:, [2, 1, 0]] * float(voxel_size_um)


def extract_native_surface_arrays(
    pore_mask: np.ndarray,
    *,
    voxel_size_um: float,
    max_surface_triangles: int,
    max_region_markers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if pore_mask.ndim != 3:
        raise ValueError(f"pore_mask must be 3D, got shape {pore_mask.shape}.")
    if not np.any(pore_mask):
        raise ValueError("No pore voxels are available for native pyGIMLi meshing.")

    padded = np.pad(pore_mask.astype(np.uint8, copy=False), 1, mode="constant", constant_values=0)
    verts_zyx, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(float(voxel_size_um), float(voxel_size_um), float(voxel_size_um)),
    )
    if faces.shape[0] > int(max_surface_triangles):
        message = (
            f"Native pyGIMLi surface extraction produced {faces.shape[0]} surface triangles, "
            f"exceeding --native-max-surface-triangles={max_surface_triangles}. "
            "This run is paused instead of downsampling or simplifying the CT geometry."
        )
        raise NativeMeshingResourcePause(
            message,
            {
                "pause_stage": "surface_triangle_guard",
                "surface_vertices": int(verts_zyx.shape[0]),
                "surface_triangles": int(faces.shape[0]),
                "max_surface_triangles": int(max_surface_triangles),
                "downsampling": "none",
                "geometry_simplification": "none",
            },
        )

    verts_zyx -= float(voxel_size_um)
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    region_markers = _component_region_marker_positions(
        pore_mask,
        float(voxel_size_um),
        max_markers=int(max_region_markers),
    )
    stats = {
        "surface_vertices": int(verts_xyz.shape[0]),
        "surface_triangles": int(faces.shape[0]),
        "region_markers": int(region_markers.shape[0]),
    }
    return verts_xyz, faces.astype(np.int32, copy=False), region_markers, stats


def build_native_pygimli_plc(
    pore_mask: np.ndarray,
    *,
    voxel_size_um: float,
    max_surface_triangles: int,
    max_region_markers: int,
):
    if pg is None:
        raise RuntimeError("pyGIMLi is not available; cannot build a native 3D PLC.")
    verts_xyz, faces, region_markers, stats = extract_native_surface_arrays(
        pore_mask,
        voxel_size_um=voxel_size_um,
        max_surface_triangles=max_surface_triangles,
        max_region_markers=max_region_markers,
    )
    plc = pg.Mesh(dim=3)
    nodes = [plc.createNode(pos.tolist()) for pos in verts_xyz]
    for tri in faces:
        plc.createTriangleFace(nodes[int(tri[0])], nodes[int(tri[1])], nodes[int(tri[2])], marker=10)
    for marker_pos in region_markers:
        plc.addRegionMarker(marker_pos.tolist(), marker=1)
    return plc, stats


def write_tetgen_poly_from_arrays(
    poly_path: Path,
    nodes_xyz: np.ndarray,
    faces: np.ndarray,
    region_markers_xyz: np.ndarray,
    *,
    boundary_marker: int = 10,
    region_marker: int = 1,
    region_area: float = 0.0,
    chunk_size: int = 100000,
) -> dict[str, int | str]:
    poly_path = Path(poly_path)
    if poly_path.suffix.lower() != ".poly":
        poly_path = poly_path.with_suffix(".poly")
    poly_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = np.asarray(nodes_xyz, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    regions = np.asarray(region_markers_xyz, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes_xyz must have shape (n, 3).")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3).")
    if regions.ndim != 2 or regions.shape[1] != 3:
        raise ValueError("region_markers_xyz must have shape (r, 3).")

    with poly_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{nodes.shape[0]}\t3\t0\t1\n")
        for start in range(0, nodes.shape[0], chunk_size):
            stop = min(start + chunk_size, nodes.shape[0])
            lines = [
                f"{idx}\t{nodes[idx, 0]:.12g}\t{nodes[idx, 1]:.12g}\t{nodes[idx, 2]:.12g}\t0\n"
                for idx in range(start, stop)
            ]
            handle.writelines(lines)

        handle.write(f"{tris.shape[0]}\t1\n")
        for start in range(0, tris.shape[0], chunk_size):
            stop = min(start + chunk_size, tris.shape[0])
            lines: list[str] = []
            for tri in tris[start:stop]:
                lines.append(f"1\t0\t{int(boundary_marker)}\n")
                lines.append(f"3\t{int(tri[0])}\t{int(tri[1])}\t{int(tri[2])}\n")
            handle.writelines(lines)

        handle.write("0\n")
        handle.write(f"{regions.shape[0]}\n")
        for idx, point in enumerate(regions):
            handle.write(
                f"{idx}\t{point[0]:.12g}\t{point[1]:.12g}\t{point[2]:.12g}\t"
                f"{int(region_marker)}\t{float(region_area):.12g}\n"
            )
    return {
        "poly_path": str(poly_path.resolve()),
        "poly_nodes": int(nodes.shape[0]),
        "poly_facets": int(tris.shape[0]),
        "poly_region_markers": int(regions.shape[0]),
    }


def parse_tetgen_statistics(text: str) -> dict[str, int]:
    label_map = {
        "Input points": "input_points",
        "Input facets": "input_facets",
        "Input segments": "input_segments",
        "Input holes": "input_holes",
        "Input regions": "input_regions",
        "Mesh points": "mesh_points",
        "Mesh tetrahedra": "mesh_tetrahedra",
        "Mesh faces": "mesh_faces",
        "Mesh faces on exterior boundary": "mesh_faces_on_exterior_boundary",
        "Mesh faces on input facets": "mesh_faces_on_input_facets",
        "Mesh edges on input segments": "mesh_edges_on_input_segments",
        "Steiner points on input facets": "steiner_points_on_input_facets",
        "Steiner points on input segments": "steiner_points_on_input_segments",
        "Steiner points inside domain": "steiner_points_inside_domain",
    }
    stats: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip()
        if label not in label_map:
            continue
        value_text = value.strip().split()[0]
        try:
            stats[label_map[label]] = int(value_text)
        except ValueError:
            continue
    return stats


def estimate_fem_resource_usage(*, mesh_points: int, mesh_tetrahedra: int) -> dict[str, int | float | str]:
    coo_entries = int(mesh_tetrahedra) * 16
    nodes_xyz_bytes = int(mesh_points) * 3 * np.dtype(np.float64).itemsize
    tets_bytes = int(mesh_tetrahedra) * 4 * np.dtype(np.int32).itemsize
    mass_and_solver_vectors_bytes = int(mesh_points) * 6 * np.dtype(np.float64).itemsize
    coo_stiffness_bytes = coo_entries * (
        2 * np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize
    )
    assembly_lower_bound_bytes = (
        nodes_xyz_bytes
        + tets_bytes
        + mass_and_solver_vectors_bytes
        + coo_stiffness_bytes
    )
    return {
        "basis": (
            "Lower bound for the current Python FEM path: nodes_xyz array, tetra index array, "
            "COO stiffness staging arrays, and six node-length float64 vectors. This excludes "
            "pyGIMLi mesh object overhead, CSR conversion overhead, boundary arrays, and CG temporaries."
        ),
        "mesh_points": int(mesh_points),
        "mesh_tetrahedra": int(mesh_tetrahedra),
        "coo_stiffness_entries": int(coo_entries),
        "nodes_xyz_bytes": int(nodes_xyz_bytes),
        "tets_bytes": int(tets_bytes),
        "mass_and_solver_vectors_bytes": int(mass_and_solver_vectors_bytes),
        "coo_stiffness_bytes": int(coo_stiffness_bytes),
        "assembly_lower_bound_bytes": int(assembly_lower_bound_bytes),
        "assembly_lower_bound_gb": float(assembly_lower_bound_bytes / 1024**3),
    }


def call_tetgen_poly(
    poly_path: Path,
    *,
    tetgen_executable: str,
    quality: float,
    max_cell_volume_um3: float,
) -> tuple[Path, str, dict[str, int]]:
    poly_path = poly_path.resolve()
    filebody = str(poly_path.with_suffix(""))
    switches = "-pzACa"
    if max_cell_volume_um3 > 0:
        switches += str(float(max_cell_volume_um3))
    switches += f"q{float(quality)}"
    command = [tetgen_executable, switches, poly_path.name]
    completed = subprocess.run(
        command,
        cwd=str(poly_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(f"TetGen failed with exit code {completed.returncode}.\n{output}")
    return Path(filebody + ".1"), output, parse_tetgen_statistics(output)


def _boundary_triangles_from_pygimli_mesh(mesh) -> np.ndarray:
    triangles = []
    for boundary in mesh.boundaries():
        node_ids = [node.id() for node in boundary.nodes()]
        if len(node_ids) == 3:
            triangles.append(node_ids)
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _tetra_cells_from_pygimli_mesh(mesh) -> np.ndarray:
    tets = []
    for cell in mesh.cells():
        node_ids = [node.id() for node in cell.nodes()]
        if len(node_ids) == 4:
            tets.append(node_ids)
    if not tets:
        raise ValueError("pyGIMLi native meshing returned no tetrahedral cells.")
    return np.asarray(tets, dtype=np.int32)


def _nodes_from_pygimli_mesh(mesh) -> np.ndarray:
    return np.asarray([[node.pos().x(), node.pos().y(), node.pos().z()] for node in mesh.nodes()], dtype=float)


def _split_external_boundary_triangles(
    nodes_xyz: np.ndarray,
    triangles: np.ndarray,
    volume_shape_zyx: tuple[int, int, int],
    voxel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    if triangles.size == 0:
        return triangles, triangles
    domain_max = np.asarray(
        [
            volume_shape_zyx[2] * voxel_size_um,
            volume_shape_zyx[1] * voxel_size_um,
            volume_shape_zyx[0] * voxel_size_um,
        ],
        dtype=float,
    )
    pts = nodes_xyz[triangles]
    tol = max(float(voxel_size_um) * 1e-6, 1e-9)
    on_min = np.any(np.all(np.abs(pts - 0.0) <= tol, axis=1), axis=1)
    on_max = np.any(np.all(np.abs(pts - domain_max[None, None, :]) <= tol, axis=1), axis=1)
    external = on_min | on_max
    return triangles[~external], triangles[external]


def create_native_pygimli_tetra_mesh(
    pore_mask: np.ndarray,
    *,
    voxel_size_um: float,
    external_boundary: str,
    tetgen_executable: str,
    quality: float,
    max_cell_volume_um3: float,
    max_surface_triangles: int,
    max_region_markers: int,
    max_tetrahedra_before_read: int,
    work_dir: Path | None = None,
) -> tuple[VoxelTetraMesh, dict[str, object]]:
    assert_native_pygimli_mesher_available(tetgen_executable=tetgen_executable)
    nodes_xyz_surface, faces, region_markers, plc_stats = extract_native_surface_arrays(
        pore_mask,
        voxel_size_um=voxel_size_um,
        max_surface_triangles=max_surface_triangles,
        max_region_markers=max_region_markers,
    )
    if work_dir is None:
        work_path = Path(tempfile.mkdtemp(prefix="pygimli_native_tetgen_"))
    else:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
    poly_path = work_path / "native_pore_surface.poly"
    poly_stats = write_tetgen_poly_from_arrays(
        poly_path,
        nodes_xyz_surface,
        faces,
        region_markers,
        region_area=0.0,
    )
    tetgen_base, tetgen_output, tetgen_stats = call_tetgen_poly(
        poly_path,
        tetgen_executable=tetgen_executable,
        quality=float(quality),
        max_cell_volume_um3=float(max_cell_volume_um3),
    )
    (work_path / "native_tetgen_stdout.log").write_text(tetgen_output, encoding="utf-8")
    mesh_tetrahedra = int(tetgen_stats.get("mesh_tetrahedra", 0))
    if mesh_tetrahedra > int(max_tetrahedra_before_read):
        resource_estimate = estimate_fem_resource_usage(
            mesh_points=int(tetgen_stats.get("mesh_points", 0)),
            mesh_tetrahedra=mesh_tetrahedra,
        )
        raise NativeMeshingResourcePause(
            (
                f"TetGen generated {mesh_tetrahedra} tetrahedra, exceeding "
                f"--native-max-tetrahedra-before-read={max_tetrahedra_before_read}. "
                "This run is paused before reading the mesh into pyGIMLi."
            ),
            {
                "pause_stage": "tetgen_tetrahedra_guard",
                "mesh_tetrahedra": mesh_tetrahedra,
                "max_tetrahedra_before_read": int(max_tetrahedra_before_read),
                "downsampling": "none",
                "geometry_simplification": "none",
                "tetgen_stats": tetgen_stats,
                "tetgen_base": str(tetgen_base.resolve()),
                "fem_resource_estimate": resource_estimate,
                **poly_stats,
            },
        )
    native_mesh = mt.readTetgen(str(tetgen_base))
    native_mesh.createNeighborInfos()
    nodes_xyz = _nodes_from_pygimli_mesh(native_mesh)
    tets = _tetra_cells_from_pygimli_mesh(native_mesh)
    all_boundary = _boundary_triangles_from_pygimli_mesh(native_mesh)
    solid_boundary, external_boundary_tris = _split_external_boundary_triangles(
        nodes_xyz,
        all_boundary,
        tuple(int(v) for v in pore_mask.shape),
        float(voxel_size_um),
    )
    solid_lumped = _lump_triangle_area(nodes_xyz, solid_boundary, nodes_xyz.shape[0])
    external_lumped = (
        _lump_triangle_area(nodes_xyz, external_boundary_tris, nodes_xyz.shape[0])
        if external_boundary == "relax"
        else np.zeros(nodes_xyz.shape[0], dtype=float)
    )
    tetra_mesh = VoxelTetraMesh(
        nodes_xyz_um=nodes_xyz,
        tets=tets,
        voxel_corner_ids=np.empty((0, 8), dtype=np.int32),
        open_voxel_coords_zyx=np.argwhere(pore_mask).astype(np.int32, copy=False),
        boundary_triangles_all=all_boundary,
        solid_boundary_triangles=solid_boundary,
        external_boundary_triangles=external_boundary_tris,
        solid_boundary_area_lumped=solid_lumped,
        external_boundary_area_lumped=external_lumped,
        solid_boundary_faces=int(solid_boundary.shape[0]),
        external_boundary_faces=int(external_boundary_tris.shape[0]),
    )
    stats = {
        **plc_stats,
        "native_mesh_nodes": int(nodes_xyz.shape[0]),
        "native_mesh_tetrahedra": int(tets.shape[0]),
        "native_mesh_boundaries": int(all_boundary.shape[0]),
        "tetgen_executable": tetgen_executable,
        "native_quality": float(quality),
        "native_max_cell_volume_um3": float(max_cell_volume_um3),
        **poly_stats,
        "tetgen_stats": tetgen_stats,
    }
    return tetra_mesh, stats


def build_voxel_tetra_mesh(
    open_mask: np.ndarray,
    *,
    voxel_size_um: float,
    external_boundary: str = "no_flux",
) -> VoxelTetraMesh:
    if open_mask.ndim != 3:
        raise ValueError(f"open_mask must be 3D, got shape {open_mask.shape}.")
    if voxel_size_um <= 0:
        raise ValueError("voxel_size_um must be positive.")
    if external_boundary not in {"no_flux", "relax"}:
        raise ValueError("external_boundary must be 'no_flux' or 'relax'.")

    coords = np.argwhere(open_mask).astype(np.int32, copy=False)
    if coords.size == 0:
        raise ValueError("No open pore voxels are available for tetrahedral meshing.")

    corners = coords[:, None, :] + CORNER_OFFSETS_ZYX[None, :, :]
    unique_corners, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    voxel_corner_ids = inverse.reshape(coords.shape[0], 8).astype(np.int32, copy=False)

    # Convert array indices z,y,x into physical x,y,z coordinates.
    nodes_xyz = unique_corners[:, [2, 1, 0]].astype(np.float64, copy=False) * float(voxel_size_um)
    tets = voxel_corner_ids[:, TET_PATTERN].reshape(-1, 4).astype(np.int32, copy=False)
    all_boundary, solid_boundary, external_boundary_tris, solid_faces, external_faces = _collect_boundary_triangles(
        open_mask,
        coords,
        voxel_corner_ids,
        include_external=True,
    )
    solid_lumped = _lump_triangle_area(nodes_xyz, solid_boundary, nodes_xyz.shape[0])
    external_lumped = (
        _lump_triangle_area(nodes_xyz, external_boundary_tris, nodes_xyz.shape[0])
        if external_boundary == "relax"
        else np.zeros(nodes_xyz.shape[0], dtype=np.float64)
    )

    return VoxelTetraMesh(
        nodes_xyz_um=nodes_xyz,
        tets=tets,
        voxel_corner_ids=voxel_corner_ids,
        open_voxel_coords_zyx=coords,
        boundary_triangles_all=all_boundary,
        solid_boundary_triangles=solid_boundary,
        external_boundary_triangles=external_boundary_tris,
        solid_boundary_area_lumped=solid_lumped,
        external_boundary_area_lumped=external_lumped,
        solid_boundary_faces=solid_faces,
        external_boundary_faces=external_faces,
    )


def create_pygimli_mesh(tetra_mesh: VoxelTetraMesh, *, progress_interval: int = 100000):
    if pg is None:
        raise RuntimeError("pyGIMLi is not available; cannot export a pyGIMLi tetra mesh.")
    mesh = pg.Mesh(dim=3)
    nodes = [mesh.createNode(pos.tolist()) for pos in tetra_mesh.nodes_xyz_um]
    for i, tet in enumerate(tetra_mesh.tets):
        mesh.createTetrahedron(nodes[int(tet[0])], nodes[int(tet[1])], nodes[int(tet[2])], nodes[int(tet[3])], marker=1)
        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            print(f"  pyGIMLi cells created: {i + 1}/{tetra_mesh.tets.shape[0]}", flush=True)
    mesh.createNeighborInfos()
    return mesh


def export_pygimli_mesh(mesh, bms_path: Path, vtk_path: Path) -> dict[str, str | bool]:
    outputs: dict[str, str | bool] = {
        "bms_written": False,
        "vtk_written": False,
        "bms_path": str(bms_path.resolve()),
        "vtk_path": str(vtk_path.resolve()),
    }
    mesh.save(str(bms_path))
    outputs["bms_written"] = bms_path.exists() and bms_path.stat().st_size > 0
    try:
        mesh.exportVTK(str(vtk_path))
        outputs["vtk_written"] = vtk_path.exists() and vtk_path.stat().st_size > 0
    except Exception as exc:  # pragma: no cover - export method depends on pygimli build
        outputs["vtk_error"] = f"{type(exc).__name__}: {exc}"
    return outputs


def assemble_tetra_fem_system(tetra_mesh: VoxelTetraMesh, *, chunk_size: int = 100000) -> TetraFemSystem:
    nodes = tetra_mesh.nodes_xyz_um
    tets = tetra_mesh.tets
    node_count = nodes.shape[0]
    mass = np.zeros(node_count, dtype=np.float64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    for start in range(0, tets.shape[0], chunk_size):
        stop = min(start + chunk_size, tets.shape[0])
        tet_chunk = tets[start:stop]
        pts = nodes[tet_chunk]
        a = np.empty((pts.shape[0], 4, 4), dtype=np.float64)
        a[:, :, 0] = 1.0
        a[:, :, 1:] = pts
        inv_a = np.linalg.inv(a)
        grads = np.transpose(inv_a[:, 1:, :], (0, 2, 1))
        det = np.linalg.det(a)
        volumes = np.abs(det) / 6.0
        if np.any(volumes <= 0.0):
            raise ValueError("Degenerate tetrahedra were found during FEM assembly.")

        np.add.at(mass, tet_chunk.ravel(), np.repeat(volumes / 4.0, 4))
        local_k = volumes[:, None, None] * np.einsum("tik,tjk->tij", grads, grads)
        rows.append(np.repeat(tet_chunk, 4, axis=1).reshape(-1))
        cols.append(np.tile(tet_chunk, (1, 4)).reshape(-1))
        data.append(local_k.reshape(-1))

    row_arr = np.concatenate(rows).astype(np.int32, copy=False)
    col_arr = np.concatenate(cols).astype(np.int32, copy=False)
    data_arr = np.concatenate(data)
    stiffness = coo_matrix((data_arr, (row_arr, col_arr)), shape=(node_count, node_count)).tocsr()
    return TetraFemSystem(
        stiffness=stiffness,
        mass_lumped=mass,
        solid_boundary_lumped=tetra_mesh.solid_boundary_area_lumped,
        external_boundary_lumped=tetra_mesh.external_boundary_area_lumped,
        node_count=node_count,
        tetra_count=tets.shape[0],
        stiffness_nonzeros=int(stiffness.nnz),
    )


def solve_tetra_decay(
    system: TetraFemSystem,
    params: TetraNmrParams,
    *,
    cg_rtol: float = 1e-6,
    cg_maxiter: int = 500,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if params.external_boundary not in {"no_flux", "relax"}:
        raise ValueError("external_boundary must be 'no_flux' or 'relax'.")

    dt = float(params.dt_ms)
    times = np.arange(0.0, params.t_max_ms + 0.5 * dt, dt)
    mass = system.mass_lumped
    diagonal = (
        mass
        + mass * (dt / float(params.bulk_t2_ms))
        + float(params.rho_solid_um_per_ms) * dt * system.solid_boundary_lumped
        + float(params.rho_external_um_per_ms) * dt * system.external_boundary_lumped
    )
    lhs = diags(diagonal, format="csr") + system.stiffness * (float(params.diffusion_um2_per_ms) * dt)
    preconditioner = LinearOperator(lhs.shape, matvec=lambda x: x / np.maximum(lhs.diagonal(), 1e-30), dtype=np.float64)
    magnetization = np.ones(system.node_count, dtype=np.float64)
    signal = np.empty(times.size, dtype=np.float64)
    rows = []

    for i, t_ms in enumerate(times):
        signal[i] = float(np.dot(mass, magnetization))
        if i == times.size - 1:
            rows.append({"time_index": i, "time_ms": float(t_ms), "cg_info": 0, "cg_iterations": 0})
            break
        iterations = 0

        def count_iteration(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        rhs = mass * magnetization
        next_magnetization, info = cg(
            lhs,
            rhs,
            x0=magnetization,
            rtol=float(cg_rtol),
            atol=0.0,
            maxiter=int(cg_maxiter),
            M=preconditioner,
            callback=count_iteration,
        )
        rows.append({"time_index": i, "time_ms": float(t_ms), "cg_info": int(info), "cg_iterations": int(iterations)})
        if info != 0:
            raise RuntimeError(f"Tetrahedral FEM CG solve failed at time index {i} with info={info}.")
        magnetization = next_magnetization

    return times, signal, pd.DataFrame(rows)


def tetra_quality_summary(nodes_xyz: np.ndarray, tets: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pts = nodes_xyz[tets]
    edge_pairs = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int32)
    edges = pts[:, edge_pairs[:, 0]] - pts[:, edge_pairs[:, 1]]
    lengths = np.linalg.norm(edges, axis=2)
    min_edge = np.min(lengths, axis=1)
    max_edge = np.max(lengths, axis=1)
    a = np.empty((pts.shape[0], 4, 4), dtype=np.float64)
    a[:, :, 0] = 1.0
    a[:, :, 1:] = pts
    volume = np.abs(np.linalg.det(a)) / 6.0
    aspect = max_edge / np.maximum(min_edge, 1e-30)
    summary = pd.DataFrame(
        [
            {
                "tetra_count": int(tets.shape[0]),
                "volume_min_um3": float(np.min(volume)),
                "volume_mean_um3": float(np.mean(volume)),
                "volume_max_um3": float(np.max(volume)),
                "edge_min_um": float(np.min(min_edge)),
                "edge_mean_um": float(np.mean(lengths)),
                "edge_max_um": float(np.max(max_edge)),
                "aspect_min": float(np.min(aspect)),
                "aspect_mean": float(np.mean(aspect)),
                "aspect_max": float(np.max(aspect)),
            }
        ]
    )
    sample_count = min(10000, tets.shape[0])
    sample_idx = np.linspace(0, tets.shape[0] - 1, sample_count, dtype=np.int64)
    sample = pd.DataFrame(
        {
            "tetra_index": sample_idx,
            "volume_um3": volume[sample_idx],
            "min_edge_um": min_edge[sample_idx],
            "max_edge_um": max_edge[sample_idx],
            "aspect_max_over_min": aspect[sample_idx],
        }
    )
    return summary, sample


def save_tetra_mesh_preview(
    nodes_xyz: np.ndarray,
    boundary_triangles: np.ndarray,
    output_path: Path,
    *,
    max_triangles: int = 25000,
) -> None:
    if boundary_triangles.size == 0:
        raise ValueError("No boundary triangles are available for mesh preview.")
    if boundary_triangles.shape[0] > max_triangles:
        pick = np.linspace(0, boundary_triangles.shape[0] - 1, max_triangles, dtype=np.int64)
        triangles = boundary_triangles[pick]
        title_suffix = f"sampled {max_triangles:,}/{boundary_triangles.shape[0]:,} boundary triangles"
    else:
        triangles = boundary_triangles
        title_suffix = f"{boundary_triangles.shape[0]:,} boundary triangles"

    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    polys = nodes_xyz[triangles]
    collection = Poly3DCollection(polys, linewidths=0.08, alpha=0.34)
    collection.set_facecolor("#7a8793")
    collection.set_edgecolor("#1f2937")
    ax.add_collection3d(collection)
    mins = np.min(nodes_xyz, axis=0)
    maxs = np.max(nodes_xyz, axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(np.maximum(maxs - mins, 1e-9))
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    ax.set_title(f"Sample 89 tetra mesh boundary preview ({title_suffix})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_tetra_mesh_geometry(tetra_mesh: VoxelTetraMesh, output_path: Path) -> None:
    np.savez_compressed(
        output_path,
        nodes_xyz_um=tetra_mesh.nodes_xyz_um,
        boundary_triangles_all=tetra_mesh.boundary_triangles_all,
        solid_boundary_triangles=tetra_mesh.solid_boundary_triangles,
        external_boundary_triangles=tetra_mesh.external_boundary_triangles,
    )


def save_decay_plot(time_ms: np.ndarray, normalized_signal: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(time_ms, normalized_signal, color="black", lw=2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized signal")
    ax.set_title("pyGIMLi tetra 3D T2 decay")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def mesh_summary_frame(
    detection: OpenPoreDetectionResult,
    tetra_mesh: VoxelTetraMesh,
    fem_system: TetraFemSystem | None,
    params: TetraNmrParams,
    build_timings: dict[str, float],
    *,
    pore_domain: str,
) -> pd.DataFrame:
    row = {
        "pore_domain": pore_domain,
        "responsive_pore_voxels": int(tetra_mesh.open_voxel_coords_zyx.shape[0]),
        "total_pore_voxels": int(detection.total_pore_voxels),
        "open_voxels": int(detection.open_pore_voxels),
        "closed_pore_voxels": int(detection.closed_pore_voxels),
        "node_count": int(tetra_mesh.nodes_xyz_um.shape[0]),
        "tetra_cell_count": int(tetra_mesh.tets.shape[0]),
        "tetra_per_open_voxel": float(tetra_mesh.tets.shape[0] / max(1, detection.open_pore_voxels)),
        "all_boundary_triangles": int(tetra_mesh.boundary_triangles_all.shape[0]),
        "solid_boundary_triangles": int(tetra_mesh.solid_boundary_triangles.shape[0]),
        "external_boundary_triangles": int(tetra_mesh.external_boundary_triangles.shape[0]),
        "solid_boundary_voxel_faces": int(tetra_mesh.solid_boundary_faces),
        "external_boundary_voxel_faces": int(tetra_mesh.external_boundary_faces),
        "voxel_size_um": float(params.voxel_size_um),
        "external_boundary": params.external_boundary,
        "stiffness_nonzeros": int(fem_system.stiffness_nonzeros) if fem_system is not None else np.nan,
    }
    row.update({f"timing_{key}_s": float(value) for key, value in build_timings.items()})
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_150_TIFF)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_89_pygimli_tetra_150"))
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument(
        "--pore-domain",
        choices=["open", "all"],
        default="all",
        help="'all' keeps every saturated pore voxel; 'open' removes isolated/closed pore components.",
    )
    parser.add_argument(
        "--meshing-backend",
        choices=["pygimli_native", "voxel_conformal"],
        default="pygimli_native",
        help="'pygimli_native' calls pyGIMLi/TetGen 3D meshing; 'voxel_conformal' uses the legacy voxel split.",
    )
    parser.add_argument("--tetgen-executable", default="tetgen")
    parser.add_argument("--native-quality", type=float, default=1.2)
    parser.add_argument(
        "--native-max-cell-volume-um3",
        type=float,
        default=0.0,
        help="TetGen maximum tetra volume in physical um^3; 0 lets TetGen decide from the PLC.",
    )
    parser.add_argument("--native-max-surface-triangles", type=int, default=2_000_000)
    parser.add_argument("--native-max-region-markers", type=int, default=20_000)
    parser.add_argument("--native-max-tetrahedra-before-read", type=int, default=8_000_000)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    parser.add_argument("--voxel-size-um", type=float, default=1.70)
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help=(
            "Integer 3D binary downsampling factor. factor=1 keeps the original voxels; "
            "factor>1 uses volume-preserving binary resampling and sets the working voxel size "
            "to factor * --voxel-size-um."
        ),
    )
    parser.add_argument("--max-open-voxels", type=int, default=500000)
    parser.add_argument("--diffusion-um2-per-ms", type=float, default=2.0)
    parser.add_argument("--bulk-t2-ms", type=float, default=3000.0)
    parser.add_argument("--rho-solid-um-per-ms", type=float, default=0.005)
    parser.add_argument("--rho-external-um-per-ms", type=float, default=0.0)
    parser.add_argument("--external-boundary", choices=["no_flux", "relax"], default="no_flux")
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--t-max-ms", type=float, default=1500.0)
    parser.add_argument("--cg-rtol", type=float, default=1e-6)
    parser.add_argument("--cg-maxiter", type=int, default=500)
    parser.add_argument("--fixed-alpha", type=float, default=476.4)
    parser.add_argument("--t2-min-ms", type=float, default=1e-2)
    parser.add_argument("--t2-max-ms", type=float, default=1e5)
    parser.add_argument("--t2-bins", type=int, default=200)
    parser.add_argument("--mesh-only", action="store_true", help="Export mesh/statistics without solving the T2 decay.")
    parser.add_argument("--skip-pygimli-export", action="store_true", help="Skip pyGIMLi Mesh creation/export.")
    parser.add_argument("--fem-chunk-size", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not args.input_tiff.exists():
        raise FileNotFoundError(f"Input TIFF not found: {args.input_tiff}")
    if args.meshing_backend == "pygimli_native":
        assert_native_pygimli_mesher_available(tetgen_executable=str(args.tetgen_executable))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()
    print(f"Loading 3D TIFF: {args.input_tiff}")
    volume = tifffile.imread(str(args.input_tiff))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF stack, got shape {volume.shape}.")
    original_volume_shape = tuple(int(v) for v in volume.shape)
    input_voxel_size_um = float(args.voxel_size_um)
    volume, downsampling_stats = downsample_binary_volume_preserve_pore_volume(
        volume,
        pore_value=int(args.pore_value),
        solid_value=int(args.solid_value),
        factor=int(args.downsample_factor),
        voxel_size_um=input_voxel_size_um,
    )
    working_voxel_size_um = float(downsampling_stats["coarse_voxel_size_um"])
    if int(args.downsample_factor) > 1:
        print(
            "Applied volume-preserving binary downsampling: "
            f"factor={args.downsample_factor}, working voxel size={working_voxel_size_um:g} um"
        )

    print("Detecting 3D open pore components...")
    detection = detect_open_pores(
        volume,
        pore_value=int(args.pore_value),
        solid_value=int(args.solid_value),
        connectivity=int(args.connectivity),
    )
    simulation_mask, responsive_pore_voxels = select_simulation_mask(
        volume,
        detection,
        pore_domain=str(args.pore_domain),
        pore_value=int(args.pore_value),
    )

    if responsive_pore_voxels > int(args.max_open_voxels):
        raise RuntimeError(
            f"Responsive pore voxels ({responsive_pore_voxels}) exceed --max-open-voxels={args.max_open_voxels}. "
            "Raise the guard only after checking memory/runtime."
        )

    detection_summary = save_detection_summary(
        detection,
        tuple(int(v) for v in volume.shape),
        working_voxel_size_um,
        args.output_dir / "open_closed_pore_summary.csv",
    )
    component_table(detection, working_voxel_size_um).to_csv(
        args.output_dir / "pore_component_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    projection_name = f"{args.pore_domain}_pore_3d_projection_preview.png"
    save_projection_preview(simulation_mask, args.output_dir / projection_name)

    params = TetraNmrParams(
        voxel_size_um=working_voxel_size_um,
        diffusion_um2_per_ms=float(args.diffusion_um2_per_ms),
        bulk_t2_ms=float(args.bulk_t2_ms),
        rho_solid_um_per_ms=float(args.rho_solid_um_per_ms),
        rho_external_um_per_ms=float(args.rho_external_um_per_ms),
        dt_ms=float(args.dt_ms),
        t_max_ms=float(args.t_max_ms),
        external_boundary=str(args.external_boundary),
    )

    timings: dict[str, float] = {}
    tic = time.perf_counter()
    if args.meshing_backend == "pygimli_native":
        print("Building native pyGIMLi/TetGen tetrahedral mesh...")
        try:
            tetra_mesh, native_mesh_stats = create_native_pygimli_tetra_mesh(
                simulation_mask,
                voxel_size_um=params.voxel_size_um,
                external_boundary=params.external_boundary,
                tetgen_executable=str(args.tetgen_executable),
                quality=float(args.native_quality),
                max_cell_volume_um3=float(args.native_max_cell_volume_um3),
                max_surface_triangles=int(args.native_max_surface_triangles),
                max_region_markers=int(args.native_max_region_markers),
                max_tetrahedra_before_read=int(args.native_max_tetrahedra_before_read),
                work_dir=args.output_dir,
            )
        except NativeMeshingResourcePause as exc:
            timings["tetra_array_build"] = time.perf_counter() - tic
            timings["total"] = time.perf_counter() - start_total
            pause_manifest = {
                "run_status": "paused_resource_guard",
                "pause_reason": str(exc),
                "pause_details": exc.details,
                "input_tiff": str(args.input_tiff.resolve()),
                "source_label_convention": {
                    str(args.pore_value): "pore voxel before open/closed filtering",
                    str(args.solid_value): "solid matrix",
                },
                "original_volume_shape_zyx": list(original_volume_shape),
                "volume_shape_zyx": list(volume.shape),
                "downsample_factor": int(args.downsample_factor),
                "downsampling": str(downsampling_stats["method"]),
                "downsampling_stats": downsampling_stats,
                "geometry_simplification": "none",
                "model_dimension_note": (
                    "This run attempted native pyGIMLi 3D tetrahedral meshing on the saturated 3D pore domain. "
                    "If downsample_factor > 1, volume-preserving binary resampling was applied before meshing "
                    "and the working voxel size records the coarser physical voxel size."
                ),
                "meshing_backend": str(args.meshing_backend),
                "connectivity": int(args.connectivity),
                "pore_domain": str(args.pore_domain),
                "nmr_params": asdict(params),
                "detection_summary": detection_summary,
                "timings_s": timings,
                "outputs": {
                    "open_closed_pore_summary_csv": str((args.output_dir / "open_closed_pore_summary.csv").resolve()),
                    "pore_component_table_csv": str((args.output_dir / "pore_component_table.csv").resolve()),
                    "projection_preview_png": str((args.output_dir / projection_name).resolve()),
                },
            }
            manifest_path = args.output_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(pause_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Run paused by resource guard. Manifest written to: {manifest_path.resolve()}")
            raise
    else:
        print("Building legacy voxel-conformal tetrahedral mesh arrays...")
        tetra_mesh = build_voxel_tetra_mesh(
            simulation_mask,
            voxel_size_um=params.voxel_size_um,
            external_boundary=params.external_boundary,
        )
        native_mesh_stats = {}
    timings["tetra_array_build"] = time.perf_counter() - tic

    tic = time.perf_counter()
    print("Saving tetrahedral mesh preview...")
    save_tetra_mesh_preview(
        tetra_mesh.nodes_xyz_um,
        tetra_mesh.boundary_triangles_all,
        args.output_dir / "pygimli_tetra_mesh_preview.png",
    )
    mesh_geometry_path = args.output_dir / "pygimli_tetra_mesh_geometry.npz"
    save_tetra_mesh_geometry(tetra_mesh, mesh_geometry_path)
    timings["mesh_preview"] = time.perf_counter() - tic

    mesh_export_outputs: dict[str, str | bool] = {}
    if not args.skip_pygimli_export:
        tic = time.perf_counter()
        print("Creating pyGIMLi Mesh object and exporting mesh files...")
        pg_mesh = create_pygimli_mesh(tetra_mesh)
        mesh_export_outputs = export_pygimli_mesh(
            pg_mesh,
            args.output_dir / "pygimli_tetra_mesh.bms",
            args.output_dir / "pygimli_tetra_mesh.vtk",
        )
        timings["pygimli_mesh_export"] = time.perf_counter() - tic
    else:
        mesh_export_outputs = {"pygimli_export_skipped": True}

    fem_system: TetraFemSystem | None = None
    if not args.mesh_only:
        tic = time.perf_counter()
        print("Assembling tetrahedral FEM system...")
        fem_system = assemble_tetra_fem_system(tetra_mesh, chunk_size=int(args.fem_chunk_size))
        timings["fem_assembly"] = time.perf_counter() - tic

    summary = mesh_summary_frame(detection, tetra_mesh, fem_system, params, timings, pore_domain=str(args.pore_domain))
    summary.to_csv(args.output_dir / "pygimli_tetra_mesh_summary.csv", index=False, encoding="utf-8-sig")
    quality_summary, quality_sample = tetra_quality_summary(tetra_mesh.nodes_xyz_um, tetra_mesh.tets)
    quality_summary.to_csv(args.output_dir / "pygimli_tetra_quality_summary.csv", index=False, encoding="utf-8-sig")
    quality_sample.to_csv(args.output_dir / "pygimli_tetra_quality_sample.csv", index=False, encoding="utf-8-sig")

    solver_outputs: dict[str, str] = {}
    inversion_outputs: dict[str, str | float | int] = {}
    decay_path = args.output_dir / "pygimli_tetra_nmr_decay.csv"
    if fem_system is not None:
        tic = time.perf_counter()
        print(f"Solving tetrahedral T2 decay for {fem_system.node_count} nodes and {fem_system.tetra_count} tetrahedra...")
        time_ms, signal, cg_history = solve_tetra_decay(
            fem_system,
            params,
            cg_rtol=float(args.cg_rtol),
            cg_maxiter=int(args.cg_maxiter),
        )
        timings["tetra_decay_solve"] = time.perf_counter() - tic
        normalized = signal / max(float(signal[0]), 1e-30)
        pd.DataFrame({"time_ms": time_ms, "signal": signal, "normalized_signal": normalized}).to_csv(decay_path, index=False)
        save_decay_plot(time_ms, normalized, args.output_dir / "pygimli_tetra_nmr_decay.png")
        cg_history_path = args.output_dir / "pygimli_tetra_cg_iteration_history.csv"
        cg_history.to_csv(cg_history_path, index=False)

        inversion_cfg = NnlsConfig(
            num_bins=int(args.t2_bins),
            regularization=float(args.fixed_alpha),
            t2_min_ms=float(args.t2_min_ms),
            t2_max_ms=float(args.t2_max_ms),
        )
        inversion = invert_single_signal_nnls(time_ms, normalized, signal_name="pygimli_tetra_3d_nmr", config=inversion_cfg)
        inversion_outputs = save_inversion_outputs(inversion, args.output_dir / "pygimli_tetra_nmr")
        solver_outputs = {
            "decay_csv": str(decay_path.resolve()),
            "decay_png": str((args.output_dir / "pygimli_tetra_nmr_decay.png").resolve()),
            "cg_iteration_history_csv": str(cg_history_path.resolve()),
            "inversion_csv": inversion_outputs["spectrum_csv"],
            "inversion_png": inversion_outputs["figure_png"],
        }
    else:
        inversion_cfg = NnlsConfig(
            num_bins=int(args.t2_bins),
            regularization=float(args.fixed_alpha),
            t2_min_ms=float(args.t2_min_ms),
            t2_max_ms=float(args.t2_max_ms),
        )

    timings["total"] = time.perf_counter() - start_total
    manifest = {
        "input_tiff": str(args.input_tiff.resolve()),
        "source_label_convention": {
            str(args.pore_value): "pore voxel before open/closed filtering",
            str(args.solid_value): "solid matrix",
        },
        "original_volume_shape_zyx": list(original_volume_shape),
        "volume_shape_zyx": list(volume.shape),
        "downsample_factor": int(args.downsample_factor),
        "downsampling": str(downsampling_stats["method"]),
        "downsampling_stats": downsampling_stats,
        "model_dimension_note": (
            "This is a true 3D tetrahedral finite-element solve on the saturated 3D pore domain. "
            "If downsample_factor > 1, the input was converted to an equivalent binary lower-resolution "
            "volume with global pore-volume conservation recorded in downsampling_stats."
        ),
        "meshing_backend": str(args.meshing_backend),
        "tetra_meshing_note": (
            "pygimli_native calls pyGIMLi meshtools.createMesh on a marching-cubes PLC at the original voxel spacing. "
            "voxel_conformal is the legacy explicit fallback and is only used when requested."
        ),
        "connectivity": int(args.connectivity),
        "pore_domain": str(args.pore_domain),
        "pore_domain_note": (
            "'all' keeps every pore voxel, including isolated/closed components. "
            "'open' keeps only pore components connected to at least one external volume face."
        ),
        "nmr_params": asdict(params),
        "inversion_config": inversion_cfg.__dict__,
        "mesh_summary": summary.iloc[0].to_dict(),
        "native_mesh_stats": native_mesh_stats,
        "quality_summary": quality_summary.iloc[0].to_dict(),
        "detection_summary": detection_summary,
        "timings_s": timings,
        "outputs": {
            "mesh_summary_csv": str((args.output_dir / "pygimli_tetra_mesh_summary.csv").resolve()),
            "quality_summary_csv": str((args.output_dir / "pygimli_tetra_quality_summary.csv").resolve()),
            "quality_sample_csv": str((args.output_dir / "pygimli_tetra_quality_sample.csv").resolve()),
            "mesh_preview_png": str((args.output_dir / "pygimli_tetra_mesh_preview.png").resolve()),
            "mesh_geometry_npz": str(mesh_geometry_path.resolve()),
            "open_closed_pore_summary_csv": str((args.output_dir / "open_closed_pore_summary.csv").resolve()),
            "pore_component_table_csv": str((args.output_dir / "pore_component_table.csv").resolve()),
            "projection_preview_png": str((args.output_dir / projection_name).resolve()),
            **mesh_export_outputs,
            **solver_outputs,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
