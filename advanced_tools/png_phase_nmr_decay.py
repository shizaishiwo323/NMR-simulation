"""Run 2D T2 decay simulation directly from red/yellow/white PNG phase maps.

Expected color meaning:
- red: water/liquid-filled pore space to be simulated
- yellow: solid matrix
- white: outside region around the sample

Boundary convention for this RTSPHEM-style image:
- water-solid contacts inside the sample use solid-liquid surface relaxation
- white region touching water on the left/right sides uses gas-liquid boundary
- white region touching water on the top/bottom sides uses solid-liquid boundary

The default model is a 2D finite-difference Bloch-Torrey T2 decay approximation.
With ``--solver triangular``, PNG phase boundaries are converted to a pyGIMLi
PLC and meshed by pyGIMLi/Triangle before a P1 finite-element decay solve.
Physical parameters are command-line inputs so the script can be used as a
reproducible workflow without hiding scientific choices.
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
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import factorized
from skimage import measure

try:
    import pygimli as pg
    import pygimli.meshtools as mt
except Exception:  # pragma: no cover - optional export dependency
    pg = None
    mt = None


WATER = np.uint8(1)
SOLID = np.uint8(2)
OUTSIDE = np.uint8(0)


@dataclass(frozen=True)
class SimulationParams:
    pixel_size_x_um: float
    pixel_size_y_um: float
    diffusion_um2_per_ms: float
    bulk_t2_ms: float
    rho_solid_um_per_ms: float
    rho_gas_um_per_ms: float
    dt_ms: float
    t_max_ms: float
    max_grid_size: int | None
    solver: str
    mesh_bulk_size_um: float
    mesh_boundary_size_um: float
    mesh_max_points: int


def _parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "original", "full"}:
        return None
    return int(value)


def classify_png(rgb: np.ndarray) -> np.ndarray:
    """Classify RGB pixels by nearest expected color."""

    colors = np.asarray(
        [
            [255, 255, 255],  # outside
            [255, 0, 0],  # water
            [255, 255, 0],  # solid
        ],
        dtype=np.int32,
    )
    labels = np.asarray([OUTSIDE, WATER, SOLID], dtype=np.uint8)
    rgb32 = rgb.astype(np.int32)
    dist2 = np.sum((rgb32[..., None, :] - colors[None, None, :, :]) ** 2, axis=-1)
    return labels[np.argmin(dist2, axis=-1)]


def downsample_nearest(labels: np.ndarray, max_grid_size: int | None) -> tuple[np.ndarray, float]:
    if max_grid_size is None:
        return labels, 1.0
    scale = max(labels.shape) / float(max_grid_size)
    if scale <= 1.0:
        return labels, 1.0
    zoom = 1.0 / scale
    small = ndimage.zoom(labels, zoom=zoom, order=0)
    return small.astype(np.uint8), scale


def sample_bbox(labels: np.ndarray) -> tuple[int, int, int, int]:
    sample = labels != OUTSIDE
    if not np.any(sample):
        raise ValueError("No red/yellow sample pixels were found in the PNG.")
    rows, cols = np.where(sample)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def boundary_kind_for_outside_neighbor(
    row: int,
    col: int,
    neighbor_row: int,
    neighbor_col: int,
    bbox: tuple[int, int, int, int],
) -> str:
    """Map a white/out-of-image neighbor to the requested physical boundary."""

    r_min, r_max, c_min, c_max = bbox
    if neighbor_col < c_min or col == c_min:
        return "gas"
    if neighbor_col > c_max or col == c_max:
        return "gas"
    if neighbor_row < r_min or row == r_min:
        return "solid"
    if neighbor_row > r_max or row == r_max:
        return "solid"
    return "gas"


def build_water_operator(labels: np.ndarray, params: SimulationParams):
    water = labels == WATER
    coords = np.argwhere(water)
    if coords.shape[0] == 0:
        raise ValueError("No red water/liquid pixels were found after classification.")

    idx = -np.ones(labels.shape, dtype=np.int64)
    idx[water] = np.arange(coords.shape[0])
    bbox = sample_bbox(labels)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    solid_surface_over_volume = np.zeros(coords.shape[0], dtype=float)
    gas_surface_over_volume = np.zeros(coords.shape[0], dtype=float)
    offsets = [
        (-1, 0, params.pixel_size_x_um, params.pixel_size_y_um**2),
        (1, 0, params.pixel_size_x_um, params.pixel_size_y_um**2),
        (0, -1, params.pixel_size_y_um, params.pixel_size_x_um**2),
        (0, 1, params.pixel_size_y_um, params.pixel_size_x_um**2),
    ]

    for k, (row, col) in enumerate(coords):
        diag = 0.0
        for dr, dc, boundary_length_um, conductance_scale_um2 in offsets:
            rr = int(row + dr)
            cc = int(col + dc)
            if 0 <= rr < labels.shape[0] and 0 <= cc < labels.shape[1]:
                neighbor = labels[rr, cc]
                if neighbor == WATER:
                    rows.append(k)
                    cols.append(int(idx[rr, cc]))
                    data.append(-1.0 / conductance_scale_um2)
                    diag += 1.0 / conductance_scale_um2
                elif neighbor == SOLID:
                    solid_surface_over_volume[k] += boundary_length_um / max(
                        params.pixel_size_x_um * params.pixel_size_y_um, 1e-12
                    )
                else:
                    kind = boundary_kind_for_outside_neighbor(int(row), int(col), rr, cc, bbox)
                    if kind == "solid":
                        solid_surface_over_volume[k] += boundary_length_um / max(
                            params.pixel_size_x_um * params.pixel_size_y_um, 1e-12
                        )
                    else:
                        gas_surface_over_volume[k] += boundary_length_um / max(
                            params.pixel_size_x_um * params.pixel_size_y_um, 1e-12
                        )
            else:
                kind = boundary_kind_for_outside_neighbor(int(row), int(col), rr, cc, bbox)
                if kind == "solid":
                    solid_surface_over_volume[k] += boundary_length_um / max(
                        params.pixel_size_x_um * params.pixel_size_y_um, 1e-12
                    )
                else:
                    gas_surface_over_volume[k] += boundary_length_um / max(
                        params.pixel_size_x_um * params.pixel_size_y_um, 1e-12
                    )

        rows.append(k)
        cols.append(k)
        data.append(diag)

    graph_laplacian = csr_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0]))
    boundary_counts = {
        "water_pixels": int(coords.shape[0]),
        "solid_liquid_surface_over_volume_sum": float(np.sum(solid_surface_over_volume)),
        "gas_liquid_surface_over_volume_sum": float(np.sum(gas_surface_over_volume)),
    }
    return graph_laplacian, solid_surface_over_volume, gas_surface_over_volume, boundary_counts


def sample_label_at_xy(labels: np.ndarray, x_um: float, y_um: float, params: SimulationParams) -> int:
    col = int(np.floor(x_um / max(params.pixel_size_x_um, 1e-12)))
    row = int(np.floor(labels.shape[0] - y_um / max(params.pixel_size_y_um, 1e-12)))
    row = int(np.clip(row, 0, labels.shape[0] - 1))
    col = int(np.clip(col, 0, labels.shape[1] - 1))
    return int(labels[row, col])


def xy_to_row_col(labels: np.ndarray, x_um: float, y_um: float, params: SimulationParams) -> tuple[int, int]:
    col = int(np.floor(x_um / max(params.pixel_size_x_um, 1e-12)))
    row = int(np.floor(labels.shape[0] - y_um / max(params.pixel_size_y_um, 1e-12)))
    return int(np.clip(row, 0, labels.shape[0] - 1)), int(np.clip(col, 0, labels.shape[1] - 1))


def contour_to_xy(contour: np.ndarray, params: SimulationParams, n_rows: int) -> np.ndarray:
    x = contour[:, 1] * params.pixel_size_x_um
    y = (n_rows - contour[:, 0]) * params.pixel_size_y_um
    return np.column_stack([x, y])


def resample_polyline(points: np.ndarray, spacing_um: float) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    closed = np.linalg.norm(points[0] - points[-1]) < max(spacing_um, 1e-12)
    if not closed:
        points = np.vstack([points, points[0]])
    seg = np.diff(points, axis=0)
    lengths = np.sqrt(np.sum(seg**2, axis=1))
    total = float(np.sum(lengths))
    if total <= 0:
        return points[:1]
    targets = np.arange(0.0, total, max(spacing_um, 1e-12))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    out = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = min(idx, lengths.size - 1)
        local = 0.0 if lengths[idx] <= 0 else (target - cumulative[idx]) / lengths[idx]
        out.append(points[idx] * (1.0 - local) + points[idx + 1] * local)
    return np.asarray(out, dtype=float)


def unique_points(points: np.ndarray, precision_um: float) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2)
    scale = 1.0 / max(precision_um, 1e-9)
    keys = np.round(points * scale).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def clean_ordered_polygon(points: np.ndarray, tolerance_um: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points.reshape(0, 2)
    cleaned = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - cleaned[-1]) > tolerance_um:
            cleaned.append(point)
    out = np.asarray(cleaned, dtype=float)
    if out.shape[0] > 1 and np.linalg.norm(out[0] - out[-1]) <= tolerance_um:
        out = out[:-1]
    return out


def signed_polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def orient_polygon(points: np.ndarray, *, ccw: bool) -> np.ndarray:
    area = signed_polygon_area(points)
    if (ccw and area < 0) or ((not ccw) and area > 0):
        return points[::-1].copy()
    return points


def choose_region_marker(labels: np.ndarray, polygon_xy: np.ndarray, params: SimulationParams) -> list[float]:
    path = MplPath(polygon_xy)
    water_rows, water_cols = np.where(labels == WATER)
    if water_rows.size == 0:
        center = np.mean(polygon_xy, axis=0)
        return [float(center[0]), float(center[1])]

    xs = (water_cols + 0.5) * params.pixel_size_x_um
    ys = (labels.shape[0] - water_rows - 0.5) * params.pixel_size_y_um
    candidates = np.column_stack([xs, ys])
    x_min, y_min = np.min(polygon_xy, axis=0)
    x_max, y_max = np.max(polygon_xy, axis=0)
    bbox = (
        (candidates[:, 0] >= x_min)
        & (candidates[:, 0] <= x_max)
        & (candidates[:, 1] >= y_min)
        & (candidates[:, 1] <= y_max)
    )
    inside_idx = np.where(bbox & path.contains_points(candidates))[0]
    if inside_idx.size:
        point = candidates[int(inside_idx[inside_idx.size // 2])]
        return [float(point[0]), float(point[1])]
    center = np.mean(polygon_xy, axis=0)
    return [float(center[0]), float(center[1])]


def pg_mesh_to_arrays(mesh, labels: np.ndarray | None = None, params: SimulationParams | None = None) -> dict[str, object]:
    raw_points = np.asarray([[node.pos().x(), node.pos().y()] for node in mesh.nodes()], dtype=float)
    raw_triangles = []
    discarded_nonwater_cells = 0
    for cell in mesh.cells():
        node_ids = [node.id() for node in cell.nodes()]
        if len(node_ids) == 3:
            if labels is not None and params is not None:
                centroid = np.mean(raw_points[node_ids], axis=0)
                if sample_label_at_xy(labels, float(centroid[0]), float(centroid[1]), params) != WATER:
                    discarded_nonwater_cells += 1
                    continue
            raw_triangles.append(node_ids)
    if not raw_triangles:
        raise ValueError("pyGIMLi/Triangle returned no triangular cells.")
    raw_triangles_arr = np.asarray(raw_triangles, dtype=np.int64)
    used_nodes = np.unique(raw_triangles_arr.ravel())
    node_map = -np.ones(raw_points.shape[0], dtype=np.int64)
    node_map[used_nodes] = np.arange(used_nodes.size)
    return {
        "points": raw_points[used_nodes],
        "triangles": node_map[raw_triangles_arr],
        "pygimli_mesh": mesh,
        "unused_pygimli_nodes_dropped": int(raw_points.shape[0] - used_nodes.size),
        "discarded_nonwater_cells": int(discarded_nonwater_cells),
    }


def build_triangular_water_mesh(labels: np.ndarray, params: SimulationParams) -> dict[str, object]:
    if mt is None:
        raise RuntimeError("pyGIMLi meshtools are not available; cannot use --solver triangular.")
    water = labels == WATER
    if not np.any(water):
        raise ValueError("No red water/liquid pixels were found after classification.")

    contours = measure.find_contours(water.astype(float), 0.5)
    polygons = []
    for contour in contours:
        xy = contour_to_xy(contour, params, labels.shape[0])
        xy = resample_polyline(xy, params.mesh_boundary_size_um)
        xy = clean_ordered_polygon(xy, tolerance_um=params.mesh_boundary_size_um * 0.05)
        if xy.shape[0] < 3:
            continue
        area = signed_polygon_area(xy)
        if abs(area) < 1e-9:
            continue
        polygons.append((area, xy))

    if not polygons:
        raise ValueError("No closed water-domain PLC contours were extracted from the PNG.")

    max_area = max(params.mesh_bulk_size_um**2 * np.sqrt(3.0) / 4.0, 1e-12)
    plcs = []
    outer_count = 0
    hole_count = 0
    for area, xy in polygons:
        if area < 0:
            outer = orient_polygon(xy, ccw=True)
            marker_pos = choose_region_marker(labels, outer, params)
            plcs.append(
                mt.createPolygon(
                    outer.tolist(),
                    isClosed=True,
                    marker=1,
                    markerPosition=marker_pos,
                    area=max_area,
                    boundaryMarker=10,
                )
            )
            outer_count += 1
        else:
            hole = orient_polygon(xy, ccw=True)
            marker_pos = [float(np.mean(hole[:, 0])), float(np.mean(hole[:, 1]))]
            plcs.append(
                mt.createPolygon(
                    hole.tolist(),
                    isClosed=True,
                    marker=0,
                    markerPosition=marker_pos,
                    area=max_area,
                    boundaryMarker=20,
                )
            )
            hole_count += 1

    if outer_count == 0:
        raise ValueError("No outer water-domain PLC contour was detected.")

    plc = mt.mergePLC(plcs, tol=max(params.mesh_boundary_size_um * 0.05, 1e-6))
    mesh = mt.createMesh(
        plc,
        quality=0,
        area=max_area,
        smooth=[1, 4],
        preserveBoundary=True,
        verbose=False,
    )
    if mesh.nodeCount() > params.mesh_max_points:
        raise ValueError(
            f"pyGIMLi/Triangle mesh node count {mesh.nodeCount()} exceeds --mesh-max-points={params.mesh_max_points}. "
            "Increase --mesh-bulk-size-um/--mesh-boundary-size-um or raise the guard after checking runtime."
        )
    out = pg_mesh_to_arrays(mesh, labels, params)
    out["plc_outer_contours"] = outer_count
    out["plc_hole_contours"] = hole_count
    out["triangle_max_area_um2"] = max_area
    out["triangle_quality_target_deg"] = 0
    return out


def classify_boundary_edge(labels: np.ndarray, midpoint: np.ndarray, params: SimulationParams) -> str:
    row, col = xy_to_row_col(labels, float(midpoint[0]), float(midpoint[1]), params)
    r0, r1 = max(0, row - 2), min(labels.shape[0], row + 3)
    c0, c1 = max(0, col - 2), min(labels.shape[1], col + 3)
    local = labels[r0:r1, c0:c1]
    if np.any(local == SOLID):
        return "solid"

    bbox = sample_bbox(labels)
    r_min, r_max, c_min, c_max = bbox
    if col <= c_min + 2 or col >= c_max - 2:
        return "gas"
    if row <= r_min + 2 or row >= r_max - 2:
        return "solid"
    return "gas"


def assemble_triangular_operator(labels: np.ndarray, params: SimulationParams):
    mesh = build_triangular_water_mesh(labels, params)
    points = mesh["points"]
    triangles = mesh["triangles"]
    n = points.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    mass = np.zeros(n, dtype=float)
    solid_robin = np.zeros(n, dtype=float)
    gas_robin = np.zeros(n, dtype=float)
    edge_counts: dict[tuple[int, int], int] = {}

    for tri in triangles:
        coords = points[tri]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area <= 1e-18:
            continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)
        local_k = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for i_local, i_global in enumerate(tri):
            mass[i_global] += area / 3.0
            for j_local, j_global in enumerate(tri):
                rows.append(int(i_global))
                cols.append(int(j_global))
                data.append(float(local_k[i_local, j_local]))
        for a, bidx in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted((int(a), int(bidx))))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    solid_length = 0.0
    gas_length = 0.0
    for (a, bidx), count in edge_counts.items():
        if count != 1:
            continue
        midpoint = (points[a] + points[bidx]) / 2.0
        length = float(np.linalg.norm(points[a] - points[bidx]))
        kind = classify_boundary_edge(labels, midpoint, params)
        if kind == "solid":
            solid_robin[a] += length / 2.0
            solid_robin[bidx] += length / 2.0
            solid_length += length
        else:
            gas_robin[a] += length / 2.0
            gas_robin[bidx] += length / 2.0
            gas_length += length

    stiffness = csr_matrix((data, (rows, cols)), shape=(n, n))
    stats = {
        "mesh_nodes": int(n),
        "mesh_triangles": int(triangles.shape[0]),
        "plc_outer_contours": int(mesh.get("plc_outer_contours", 0)),
        "plc_hole_contours": int(mesh.get("plc_hole_contours", 0)),
        "unused_pygimli_nodes_dropped": int(mesh.get("unused_pygimli_nodes_dropped", 0)),
        "triangle_max_area_um2": float(mesh.get("triangle_max_area_um2", np.nan)),
        "triangle_quality_target_deg": int(mesh.get("triangle_quality_target_deg", 32)),
        "solid_liquid_boundary_length_um": float(solid_length),
        "gas_liquid_boundary_length_um": float(gas_length),
        "mesh_bulk_size_um": float(params.mesh_bulk_size_um),
        "mesh_boundary_size_um": float(params.mesh_boundary_size_um),
    }
    return mesh, stiffness, mass, solid_robin, gas_robin, stats


def save_triangular_mesh_plot(mesh: dict[str, object], output_path: Path) -> None:
    points = mesh["points"]
    triangles = mesh["triangles"]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.triplot(triangulation, color="0.25", lw=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Triangular water-domain mesh")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def triangle_quality_table(mesh: dict[str, object]) -> pd.DataFrame:
    points = mesh["points"]
    triangles = mesh["triangles"]
    rows = []
    for elem_id, tri in enumerate(triangles):
        coords = points[tri]
        edge01 = float(np.linalg.norm(coords[1] - coords[0]))
        edge12 = float(np.linalg.norm(coords[2] - coords[1]))
        edge20 = float(np.linalg.norm(coords[0] - coords[2]))
        area = 0.5 * abs(
            (coords[1, 0] - coords[0, 0]) * (coords[2, 1] - coords[0, 1])
            - (coords[2, 0] - coords[0, 0]) * (coords[1, 1] - coords[0, 1])
        )
        denom = edge01**2 + edge12**2 + edge20**2
        quality = 4.0 * np.sqrt(3.0) * area / denom if denom > 0 else 0.0
        rows.append(
            {
                "element_id": elem_id,
                "node_0": int(tri[0]),
                "node_1": int(tri[1]),
                "node_2": int(tri[2]),
                "area_um2": area,
                "edge_min_um": min(edge01, edge12, edge20),
                "edge_max_um": max(edge01, edge12, edge20),
                "edge_mean_um": (edge01 + edge12 + edge20) / 3.0,
                "quality_equilateral_normalized": quality,
                "aspect_edge_ratio": max(edge01, edge12, edge20) / max(min(edge01, edge12, edge20), 1e-30),
            }
        )
    return pd.DataFrame(rows)


def mesh_boundary_edge_count(triangles: np.ndarray) -> int:
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, bidx in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted((int(a), int(bidx))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return int(sum(1 for count in edge_counts.values() if count == 1))


def summarize_mesh_quality(mesh: dict[str, object], quality_frame: pd.DataFrame) -> dict[str, float | int]:
    areas = quality_frame["area_um2"].to_numpy(dtype=float)
    qualities = quality_frame["quality_equilateral_normalized"].to_numpy(dtype=float)
    aspect = quality_frame["aspect_edge_ratio"].to_numpy(dtype=float)
    return {
        "mesh_vertices": int(mesh["points"].shape[0]),
        "mesh_triangles": int(mesh["triangles"].shape[0]),
        "boundary_edges": mesh_boundary_edge_count(mesh["triangles"]),
        "min_element_quality": float(np.min(qualities)),
        "mean_element_quality": float(np.mean(qualities)),
        "median_element_quality": float(np.median(qualities)),
        "p05_element_quality": float(np.percentile(qualities, 5.0)),
        "p95_element_quality": float(np.percentile(qualities, 95.0)),
        "min_element_area_um2": float(np.min(areas)),
        "mean_element_area_um2": float(np.mean(areas)),
        "max_element_area_um2": float(np.max(areas)),
        "element_area_ratio_min_over_max": float(np.min(areas) / max(np.max(areas), 1e-30)),
        "mesh_area_um2": float(np.sum(areas)),
        "max_aspect_edge_ratio": float(np.max(aspect)),
        "mean_aspect_edge_ratio": float(np.mean(aspect)),
    }


def save_mesh_quality_histogram(quality_frame: pd.DataFrame, output_path: Path) -> None:
    quality = quality_frame["quality_equilateral_normalized"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(quality, bins=np.linspace(0.0, 1.0, 41), color="0.1", edgecolor="0.1")
    ax.set_xlabel("element quality (0-1)")
    ax.set_ylabel("element count")
    ax.set_title("Triangular mesh quality histogram")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pygimli_mesh(mesh: dict[str, object], output_path: Path) -> bool:
    if pg is None:
        return False
    try:
        if "pygimli_mesh" in mesh:
            mesh["pygimli_mesh"].save(str(output_path))
            return True
        points = mesh["points"]
        triangles = mesh["triangles"]
        pg_mesh = pg.Mesh(2)
        nodes = [pg_mesh.createNode(float(x), float(y), 0.0) for x, y in points]
        for tri in triangles:
            pg_mesh.createTriangle(nodes[int(tri[0])], nodes[int(tri[1])], nodes[int(tri[2])])
        pg_mesh.save(str(output_path))
        return True
    except RuntimeError:
        return False


def solve_decay_triangular(
    labels: np.ndarray,
    params: SimulationParams,
    mesh_plot_path: Path | None = None,
    mesh_bms_path: Path | None = None,
    mesh_quality_csv_path: Path | None = None,
    mesh_quality_hist_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mesh, stiffness, mass, solid_robin, gas_robin, stats = assemble_triangular_operator(labels, params)
    if mesh_plot_path is not None:
        save_triangular_mesh_plot(mesh, mesh_plot_path)
    if mesh_bms_path is not None:
        stats["pygimli_bms_written"] = bool(save_pygimli_mesh(mesh, mesh_bms_path))
    quality_frame = triangle_quality_table(mesh)
    quality_summary = summarize_mesh_quality(mesh, quality_frame)
    stats.update(quality_summary)
    if mesh_quality_csv_path is not None:
        quality_frame.to_csv(mesh_quality_csv_path, index=False)
    if mesh_quality_hist_path is not None:
        save_mesh_quality_histogram(quality_frame, mesh_quality_hist_path)

    dt = params.dt_ms
    mass_matrix = diags(mass, format="csr")
    lhs = (
        mass_matrix
        + stiffness * (params.diffusion_um2_per_ms * dt)
        + mass_matrix * (dt / params.bulk_t2_ms)
        + diags(params.rho_solid_um_per_ms * dt * solid_robin, format="csr")
        + diags(params.rho_gas_um_per_ms * dt * gas_robin, format="csr")
    )
    try:
        solve = factorized(lhs.tocsc())
        stats["triangular_factorization_jitter"] = 0.0
    except RuntimeError:
        jitter = max(float(np.max(np.abs(lhs.diagonal()))), 1.0) * 1e-12
        solve = factorized((lhs + eye(lhs.shape[0], format="csr") * jitter).tocsc())
        stats["triangular_factorization_jitter"] = jitter
    times = np.arange(0.0, params.t_max_ms + 0.5 * dt, dt)
    magnetization = np.ones_like(mass, dtype=float)
    amplitude = np.empty_like(times)
    for i in range(times.size):
        amplitude[i] = float(np.dot(mass, magnetization))
        magnetization = solve(mass * magnetization)
    return times, amplitude, stats


def solve_decay_pixel(labels: np.ndarray, params: SimulationParams) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    lap, solid_sink, gas_sink, boundary_counts = build_water_operator(labels, params)
    n = lap.shape[0]
    dt = params.dt_ms
    lhs = (
        eye(n, format="csr")
        + lap * (params.diffusion_um2_per_ms * dt)
        + eye(n, format="csr") * (dt / params.bulk_t2_ms)
        + diags(params.rho_solid_um_per_ms * dt * solid_sink, format="csr")
        + diags(params.rho_gas_um_per_ms * dt * gas_sink, format="csr")
    )
    solve = factorized(lhs.tocsc())
    times = np.arange(0.0, params.t_max_ms + 0.5 * dt, dt)
    magnetization = np.ones(n, dtype=float)
    amplitude = np.empty_like(times)
    pixel_area_um2 = params.pixel_size_x_um * params.pixel_size_y_um
    for i in range(times.size):
        amplitude[i] = float(np.sum(magnetization) * pixel_area_um2)
        magnetization = solve(magnetization)
    return times, amplitude, boundary_counts


def save_decay_plot(time_ms: np.ndarray, normalized_signal: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(time_ms, normalized_signal, color="black", lw=2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized NMR signal")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def simulate_png(png_path: Path, output_dir: Path, params: SimulationParams) -> dict[str, object]:
    rgb = np.asarray(Image.open(png_path).convert("RGB"))
    labels_full = classify_png(rgb)
    labels, scale = downsample_nearest(labels_full, params.max_grid_size)
    effective_params = SimulationParams(
        pixel_size_x_um=params.pixel_size_x_um * scale,
        pixel_size_y_um=params.pixel_size_y_um * scale,
        diffusion_um2_per_ms=params.diffusion_um2_per_ms,
        bulk_t2_ms=params.bulk_t2_ms,
        rho_solid_um_per_ms=params.rho_solid_um_per_ms,
        rho_gas_um_per_ms=params.rho_gas_um_per_ms,
        dt_ms=params.dt_ms,
        t_max_ms=params.t_max_ms,
        max_grid_size=params.max_grid_size,
        solver=params.solver,
        mesh_bulk_size_um=params.mesh_bulk_size_um,
        mesh_boundary_size_um=params.mesh_boundary_size_um,
        mesh_max_points=params.mesh_max_points,
    )

    stem = png_path.stem
    curve_path = output_dir / f"{stem}_nmr_decay.csv"
    plot_path = output_dir / f"{stem}_nmr_decay.png"
    mesh_plot_path = output_dir / f"{stem}_triangular_mesh.png"
    mesh_bms_path = output_dir / f"{stem}_triangular_mesh.bms"
    mesh_quality_csv_path = output_dir / f"{stem}_mesh_quality.csv"
    mesh_quality_hist_path = output_dir / f"{stem}_mesh_quality_histogram.png"
    if effective_params.solver == "triangular":
        time_ms, amplitude, boundary_counts = solve_decay_triangular(
            labels,
            effective_params,
            mesh_plot_path,
            mesh_bms_path,
            mesh_quality_csv_path,
            mesh_quality_hist_path,
        )
    else:
        time_ms, amplitude, boundary_counts = solve_decay_pixel(labels, effective_params)
    normalized = amplitude / max(amplitude[0], 1e-30)
    pd.DataFrame(
        {
            "time_ms": time_ms,
            "signal": amplitude,
            "normalized_signal": normalized,
        }
    ).to_csv(curve_path, index=False)
    save_decay_plot(time_ms, normalized, plot_path, title=f"{stem} T2 decay")

    full_counts = {
        "outside_px": int(np.sum(labels_full == OUTSIDE)),
        "water_px": int(np.sum(labels_full == WATER)),
        "solid_px": int(np.sum(labels_full == SOLID)),
    }
    return {
        "input_path": str(png_path.resolve()),
        "curve_csv": str(curve_path.resolve()),
        "curve_png": str(plot_path.resolve()),
        "raw_shape": list(labels_full.shape),
        "simulation_shape": list(labels.shape),
        "downsample_scale": float(scale),
        "params_used": asdict(effective_params),
        "phase_counts_full_resolution": full_counts,
        "mesh_or_boundary_summary": boundary_counts,
        "mesh_png": str(mesh_plot_path.resolve()) if effective_params.solver == "triangular" else None,
        "pygimli_mesh_bms": str(mesh_bms_path.resolve()) if effective_params.solver == "triangular" else None,
        "mesh_quality_csv": str(mesh_quality_csv_path.resolve()) if effective_params.solver == "triangular" else None,
        "mesh_quality_histogram_png": (
            str(mesh_quality_hist_path.resolve()) if effective_params.solver == "triangular" else None
        ),
        "boundary_interpretation": {
            "red": "water/liquid-filled pore domain solved by the Bloch-Torrey T2 decay approximation",
            "yellow": "solid matrix; red-yellow contacts use solid-liquid surface relaxation",
            "white_left_right": "gas-liquid boundary; rho_gas is applied only if nonzero",
            "white_top_bottom": "fixed solid-liquid boundary using rho_solid",
        },
    }


def iter_input_pngs(input_path: Path, pattern: str, limit: int | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    pngs = sorted(input_path.glob(pattern))
    if limit is not None:
        pngs = pngs[:limit]
    if not pngs:
        raise FileNotFoundError(f"No PNG files matched {pattern!r} in {input_path}")
    return pngs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="PNG file or directory containing PNG phase maps.")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/png_phase_nmr_decay"))
    parser.add_argument("--pattern", default="*.png", help="Glob used when input_path is a directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of PNG files to process from a directory.")
    parser.add_argument("--pixel-size-um", type=float, default=1.0)
    parser.add_argument("--pixel-size-x-um", type=float, default=None)
    parser.add_argument("--pixel-size-y-um", type=float, default=None)
    parser.add_argument("--length-x-cm", type=float, default=None)
    parser.add_argument("--length-y-cm", type=float, default=None)
    parser.add_argument("--diffusion-um2-per-ms", type=float, default=2.0)
    parser.add_argument("--bulk-t2-ms", type=float, default=3000.0)
    parser.add_argument("--rho-solid-um-per-ms", type=float, default=0.005)
    parser.add_argument("--rho-gas-um-per-ms", type=float, default=0.0)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--t-max-ms", type=float, default=1500.0)
    parser.add_argument("--solver", choices=["pixel", "triangular"], default="pixel")
    parser.add_argument(
        "--mesh-bulk-size-um",
        type=float,
        default=8.0,
        help="Target interior point spacing for triangular meshing; smaller means finer bulk mesh.",
    )
    parser.add_argument(
        "--mesh-boundary-size-um",
        type=float,
        default=2.0,
        help="Target boundary point spacing for triangular meshing; smaller means denser water/solid and water/gas boundaries.",
    )
    parser.add_argument(
        "--mesh-max-points",
        type=int,
        default=25000,
        help="Safety guard for pyGIMLi/Triangle mesh nodes.",
    )
    parser.add_argument(
        "--max-grid-size",
        type=_parse_optional_int,
        default=500,
        help="Downsample so the largest image dimension is at most this value. Use 'none' for full resolution.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pngs = iter_input_pngs(args.input_path, args.pattern, args.limit)
    first_shape = np.asarray(Image.open(pngs[0]).convert("RGB")).shape[:2]
    if args.length_x_cm is not None:
        pixel_size_x_um = args.length_x_cm * 10000.0 / first_shape[1]
    else:
        pixel_size_x_um = args.pixel_size_x_um if args.pixel_size_x_um is not None else args.pixel_size_um
    if args.length_y_cm is not None:
        pixel_size_y_um = args.length_y_cm * 10000.0 / first_shape[0]
    else:
        pixel_size_y_um = args.pixel_size_y_um if args.pixel_size_y_um is not None else args.pixel_size_um
    params = SimulationParams(
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        diffusion_um2_per_ms=args.diffusion_um2_per_ms,
        bulk_t2_ms=args.bulk_t2_ms,
        rho_solid_um_per_ms=args.rho_solid_um_per_ms,
        rho_gas_um_per_ms=args.rho_gas_um_per_ms,
        dt_ms=args.dt_ms,
        t_max_ms=args.t_max_ms,
        max_grid_size=args.max_grid_size,
        solver=args.solver,
        mesh_bulk_size_um=args.mesh_bulk_size_um,
        mesh_boundary_size_um=args.mesh_boundary_size_um,
        mesh_max_points=args.mesh_max_points,
    )
    summaries = []
    for png_path in pngs:
        summaries.append(simulate_png(png_path, args.output_dir, params))

    summary_path = args.output_dir / "png_phase_nmr_decay_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Processed {len(summaries)} PNG file(s).")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
