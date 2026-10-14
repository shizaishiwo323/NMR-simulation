"""Ideal 2-D big pore - slit throat - small pore chain: pyGIMLi NMR, 2-D SIP and joint size figure.

Workflow
--------
1. Geometry: periodic chain ... B - throat - S - throat - B ... (2-D, per unit depth).
2. NMR: pyGIMLi triangle meshes; Robin Bloch-Torrey implicit-Euler stepping with the
   same discretization as ``pg.solve`` in ``Auto_NMR._pg_solve_step`` (verified each run),
   solved on the mirror-symmetric half cell [B/2 - throat - S/2], which is exact for the
   chain. Isolated B, S and throat meshes give the uncoupled reference.
3. T2 inversion: ``nmr_t2`` fixed-regularization NNLS and L-curve NNLS on the same noisy decay.
4. SIP: reads (or runs) the result package of
   ``sip模拟/code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py``.
5. Figures: geometry/meshes, size-selection windows, NMR results, SIP mechanisms, and the
   joint overlay of actual pore-body/throat size histograms, NMR T2 and SIP sigma''/Debye
   radius distribution.

Run from the NMR package root:

    python advanced_tools/ideal_two_pore_channel_nmr_sip.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import pygimli as pg
import pygimli.meshtools as mt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import brentq
from scipy.special import j0, j1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmr_t2.config import LCurveConfig, NnlsConfig  # noqa: E402
from nmr_t2.lcurve import invert_single_signal_lcurve  # noqa: E402
from nmr_t2.nnls import invert_single_signal_nnls  # noqa: E402

# ---------------------------------------------------------------------------
# Shared geometry (um). Must equal configs/sip_simulation/ideal_two_pore_channel_sip2d_config.py
# ---------------------------------------------------------------------------
GEOMETRY_UM = {
    "big_radius": 20.0,
    "small_radius": 5.0,
    "throat_width": 0.1,
    "throat_length": 15.0,
    "solid_margin": 2.0,
}

# NMR parameters (um, ms): Auto_NMR.py defaults (Zhou & Zhang, 2025, Table A1; rho2 = 5 um/s).
DIFFUSION_UM2_MS = 2.0
RHO2_UM_MS = 0.005
BULK_T2_MS = 3000.0
TIME_STEP_MS = 0.25  # echo spacing; implicit-Euler rate bias dt/(2 T2) < 1.3% for the 10-ms throat mode
TIME_MAX_MS = 6000.0
# Synthetic acquisition noise (Gaussian, relative to M0). SNR = 1000 represents a stacked acquisition;
# the Auto_NMR L-curve convention (1%, SNR = 100) is run as a sensitivity case.
NOISE_RELATIVE = 0.001
NOISE_SENSITIVITY = 0.01
NOISE_SEED = 20261012
FIXED_ALPHA = 1.0  # nmr_t2.NnlsConfig default regularization
# 200 log bins from 1 ms (= 4 TE, NnlsConfig default lower bound; the shortest physical mode is the
# 10-ms throat) to 1e5 ms. A 0.01-ms floor lets 1% noise on the first echoes create a spurious ~0.7-ms peak.
T2_BINS = dict(num_bins=200, t2_min_ms=1.0, t2_max_ms=1e5)
# L-curve scan for nmr_t2.invert_single_signal_lcurve (project reciprocal-slope criterion, target 0.25).
# The default scan stops at alpha = 100, below where the criterion is met for 1% noise; extended so the
# selected alpha is an interior point of the scan.
LCURVE_ALPHAS = np.logspace(-3, 5, 81)

MESH_LEVELS = {
    "fine": {"body_area_um2": 0.04, "throat_cells_across": 4, "segment_um": 0.12},
    "coarse": {"body_area_um2": 0.16, "throat_cells_across": 2, "segment_um": 0.24},
}

# Radius conversions used only for display.
SIP_RADIUS_DIFFUSION_M2_S = 3.8e-12  # user-specified D+ (Schwarz 1962; Nordsiek & Weller 2008; Zhang et al. 2018)

# Categorical slots (dataviz reference palette), fixed per entity across figures.
C_BODY, C_THROAT, C_NMR, C_SIP = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET = "#e87ba4", "#008300", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


# ---------------------------------------------------------------------------
# Geometry and pyGIMLi meshes
# ---------------------------------------------------------------------------
def arc(cx, cy, r, a0, a1, n):
    return [[cx + r * math.cos(t), cy + r * math.sin(t)] for t in np.linspace(a0, a1, n)]


def half_cell_plc(level: dict):
    """[right half of B] - throat - [left half of S]; region markers 1, 2, 3."""

    rb, rs = GEOMETRY_UM["big_radius"], GEOMETRY_UM["small_radius"]
    h, length = 0.5 * GEOMETRY_UM["throat_width"], GEOMETRY_UM["throat_length"]
    seg = level["segment_um"]
    tb, ts = math.asin(h / rb), math.asin(h / rs)
    xb, xs_center = rb * math.cos(tb), rb + length + rs
    xs = xs_center - rs * math.cos(ts)
    nb = max(24, math.ceil(rb * math.pi / 2 / seg))
    ns = max(24, math.ceil(rs * math.pi / 2 / seg))
    throat_area = (2 * h / level["throat_cells_across"]) ** 2 * math.sqrt(3) / 4
    big = mt.createPolygon(
        [[0.0, -rb]] + arc(0, 0, rb, -math.pi / 2, -tb, nb)[1:] + arc(0, 0, rb, tb, math.pi / 2, nb),
        isClosed=True, marker=1, area=level["body_area_um2"],
    )
    nt = max(8, math.ceil((xs - xb) / (2 * h / level["throat_cells_across"])))
    bottom = [[xb + (xs - xb) * k / nt, -h] for k in range(nt + 1)]
    top = [[xs - (xs - xb) * k / nt, h] for k in range(nt + 1)]
    throat = mt.createPolygon(bottom + top, isClosed=True, marker=2, area=throat_area)
    small = mt.createPolygon(
        arc(xs_center, 0, rs, math.pi + ts, 1.5 * math.pi, ns) + [[xs_center, rs]] + arc(xs_center, 0, rs, math.pi / 2, math.pi - ts, ns)[1:],
        isClosed=True, marker=3, area=level["body_area_um2"],
    )
    return mt.mergePLC([big, throat, small]), xs_center


def disk_plc(radius: float, level: dict):
    n = max(48, math.ceil(2 * math.pi * radius / level["segment_um"]))
    return mt.createPolygon(arc(0, 0, radius, 0, 2 * math.pi, n + 1)[:-1], isClosed=True, marker=1, area=level["body_area_um2"])


def throat_plc(level: dict):
    h, length = 0.5 * GEOMETRY_UM["throat_width"], GEOMETRY_UM["throat_length"]
    step = 2 * h / level["throat_cells_across"]
    nt = max(8, math.ceil(length / step))
    pts = [[length * k / nt, -h] for k in range(nt + 1)] + [[length - length * k / nt, h] for k in range(nt + 1)]
    return mt.createPolygon(pts, isClosed=True, marker=2, area=step**2 * math.sqrt(3) / 4)


def save_mesh(mesh, path: Path) -> None:
    """pyGIMLi's C++ writer cannot open non-ASCII paths (e.g. 'NMR模拟'); write to an ASCII temp file first."""

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "mesh.bms"
        mesh.save(str(staged))
        shutil.move(str(staged), str(path))


def build_mesh(plc, symmetry_x=()):
    """Mesh and mark outer boundaries: 1 = solid wall (Robin), 2 = symmetry cut (no flux)."""

    mesh = mt.createMesh(plc, quality=33)
    for boundary in mesh.boundaries():
        if not boundary.outside():
            boundary.setMarker(0)
            continue
        cx = boundary.center().x()
        on_cut = any(abs(cx - xc) < 1e-7 and abs(boundary.node(0).pos().x() - boundary.node(1).pos().x()) < 1e-9 for xc in symmetry_x)
        boundary.setMarker(2 if on_cut else 1)
    return mesh


# ---------------------------------------------------------------------------
# Bloch-Torrey stepper (same discretization as pg.solve; verified)
# ---------------------------------------------------------------------------
def p1_operators(mesh):
    nodes = np.array([[n.pos().x(), n.pos().y()] for n in mesh.nodes()])
    tri = np.array([[c.node(i).id() for i in range(3)] for c in mesh.cells()])
    p = nodes[tri]
    x, y = p[..., 0], p[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    area = 0.5 * np.abs(np.sum(x * b, axis=1))
    stiffness = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * area[:, None, None])
    mass = area[:, None, None] / 12.0 * (np.ones((3, 3)) + np.eye(3))
    rows, cols = np.repeat(tri, 3, axis=1).ravel(), np.tile(tri, (1, 3)).ravel()
    n = len(nodes)
    K = sp.coo_matrix((stiffness.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    M = sp.coo_matrix((mass.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    er, ec, ev = [], [], []
    for boundary in mesh.boundaries():
        if boundary.marker() == 1:
            i, j, le = boundary.node(0).id(), boundary.node(1).id(), boundary.size()
            er += [i, i, j, j]
            ec += [i, j, i, j]
            ev += [le / 3, le / 6, le / 6, le / 3]
    B = sp.coo_matrix((ev, (er, ec)), shape=(n, n)).tocsc()
    return K, M, B, tri, area


class BlochTorreyStepper:
    """Implicit Euler: ((1 + dt/T2B) M + dt D K + dt rho B) u_new = M_lumped u_old.

    pg.solve(mesh, a=D dt, b=-(1 + dt/T2B), f=u_old, bc={'Robin': {1: rho dt}}) assembles
    exactly this system (consistent mass on the left, lumped-mass load vector on the right);
    the factorization is computed once and reused.
    """

    def __init__(self, mesh, dt):
        self.mesh, self.dt = mesh, dt
        K, M, B, self.tri, self.cell_area = p1_operators(mesh)
        self.lumped = np.asarray(M.sum(axis=1)).ravel()
        system = (1 + dt / BULK_T2_MS) * M + dt * DIFFUSION_UM2_MS * K + dt * RHO2_UM_MS * B
        self.lu = spla.splu(system.tocsc())
        self.cell_marker = np.array([cell.marker() for cell in mesh.cells()])
        # Region integral of the P1 field: each cell contributes area/3 at its three nodes.
        self.markers = [int(m) for m in np.unique(self.cell_marker)]
        rows = np.repeat(np.searchsorted(self.markers, self.cell_marker), 3)
        self.region_weights = sp.coo_matrix(
            (np.repeat(self.cell_area / 3.0, 3), (rows, self.tri.ravel())), shape=(len(self.markers), mesh.nodeCount())
        ).tocsr()

    def step(self, u):
        return self.lu.solve(self.lumped * u)

    def verify_against_pg_solve(self, steps: int = 3) -> float:
        bc = {"Robin": {1: RHO2_UM_MS * self.dt}}
        ours = theirs = np.ones(self.mesh.nodeCount())
        worst = 0.0
        for _ in range(steps):
            ours = self.step(ours)
            theirs = np.asarray(pg.solve(self.mesh, a=DIFFUSION_UM2_MS * self.dt, b=-(1 + self.dt / BULK_T2_MS), f=theirs, bc=bc), dtype=float)
            worst = max(worst, float(np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs))))
        return worst

    def run(self, n_steps):
        u = np.ones(self.mesh.nodeCount())
        total = np.empty(n_steps + 1)
        regional = np.empty((len(self.markers), n_steps + 1))
        for k in range(n_steps + 1):
            if k:
                u = self.step(u)
            total[k] = self.lumped @ u
            regional[:, k] = self.region_weights @ u
        return total, dict(zip(self.markers, regional))


def mesh_summary(mesh) -> dict:
    areas = np.asarray(mesh.cellSizes())
    return {
        "nodes": int(mesh.nodeCount()),
        "cells": int(mesh.cellCount()),
        "area_um2": float(areas.sum()),
        "min_cell_area_um2": float(areas.min()),
        "max_cell_area_um2": float(areas.max()),
        "robin_wall_length_um": float(sum(b.size() for b in mesh.boundaries() if b.marker() == 1)),
        "symmetry_cut_length_um": float(sum(b.size() for b in mesh.boundaries() if b.marker() == 2)),
    }


# ---------------------------------------------------------------------------
# Analytic references and conversions
# ---------------------------------------------------------------------------
def bessel_disk_t2_ms(radius_um: float) -> float:
    kappa = RHO2_UM_MS * radius_um / DIFFUSION_UM2_MS
    xi = brentq(lambda v: v * j1(v) / j0(v) - kappa, 1e-12, 2.404825557 - 1e-9)
    return 1.0 / (DIFFUSION_UM2_MS * xi**2 / radius_um**2 + 1.0 / BULK_T2_MS)


def slab_t2_ms(half_width_um: float) -> float:
    kappa = RHO2_UM_MS * half_width_um / DIFFUSION_UM2_MS
    xi = brentq(lambda v: v * math.tan(v) - kappa, 1e-12, math.pi / 2 - 1e-9)
    return 1.0 / (DIFFUSION_UM2_MS * xi**2 / half_width_um**2 + 1.0 / BULK_T2_MS)


def nmr_radius_um(t2_ms):
    """Disk (circular cross-section) fast-diffusion map with bulk correction: r = 2 rho2 / (1/T2 - 1/T2B)."""

    rate = 1.0 / np.asarray(t2_ms, dtype=float) - 1.0 / BULK_T2_MS
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rate > 0, 2.0 * RHO2_UM_MS / rate, np.nan)


def nmr_t2_from_radius_ms(radius_um):
    return 1.0 / (2.0 * RHO2_UM_MS / np.asarray(radius_um, dtype=float) + 1.0 / BULK_T2_MS)


def sip_frequency_from_radius_hz(radius_um, diffusion_m2_s=SIP_RADIUS_DIFFUSION_M2_S):
    return diffusion_m2_s / (np.pi * (np.asarray(radius_um, dtype=float) * 1e-6) ** 2)


def sip_radius_from_frequency_um(frequency_hz, diffusion_m2_s=SIP_RADIUS_DIFFUSION_M2_S):
    return np.sqrt(diffusion_m2_s / (np.pi * np.asarray(frequency_hz, dtype=float))) * 1e6


def spectrum_peaks(x, amplitude, threshold=0.01):
    a = np.asarray(amplitude, dtype=float)
    if a.max() <= 0:
        return []
    idx = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > threshold * a.max()]
    edges = [0] + [idx[k] + int(np.argmin(a[idx[k] : idx[k + 1]])) for k in range(len(idx) - 1)] + [len(a)]
    x = np.asarray(x, dtype=float)
    out = []
    for k, i in enumerate(idx):
        window = slice(edges[k], edges[k + 1])
        # T2lm of this mode only: contiguous non-zero support around the peak, so that isolated
        # noise-floor amplitude elsewhere in the valley-to-valley window does not bias it.
        lo, hi = i, i
        floor = 1e-6 * a[i]
        while lo > edges[k] and a[lo - 1] > floor:
            lo -= 1
        while hi < edges[k + 1] - 1 and a[hi + 1] > floor:
            hi += 1
        out.append({
            "position": float(x[i]),
            "fraction": float(a[window].sum() / a.sum()),
            "t2lm": float(10 ** np.average(np.log10(x[lo : hi + 1]), weights=a[lo : hi + 1])),
        })
    return out


# ---------------------------------------------------------------------------
# NMR simulation and inversion
# ---------------------------------------------------------------------------
def simulate_nmr(out: Path) -> dict:
    n_steps = int(round(TIME_MAX_MS / TIME_STEP_MS))
    times = np.arange(n_steps + 1) * TIME_STEP_MS
    h = 0.5 * GEOMETRY_UM["throat_width"]
    results: dict = {"times_ms": times, "meshes": {}, "verification": {}, "timing_s": {}}
    for level_name, level in MESH_LEVELS.items():
        plc, xs_center = half_cell_plc(level)
        mesh = build_mesh(plc, symmetry_x=(0.0, xs_center))
        save_mesh(mesh, out / "source_data" / f"nmr_half_cell_mesh_{level_name}.bms")
        start = time.perf_counter()
        stepper = BlochTorreyStepper(mesh, TIME_STEP_MS)
        if level_name == "coarse":
            results["verification"]["max_rel_diff_vs_pg_solve_3_steps"] = stepper.verify_against_pg_solve(3)
        total, regional = stepper.run(n_steps)
        results["timing_s"][f"connected_{level_name}"] = time.perf_counter() - start
        results["meshes"][f"connected_{level_name}"] = {"mesh": mesh, **mesh_summary(mesh)}
        # Half cell -> one period of the chain (mirror symmetry doubles every region).
        results[f"connected_{level_name}"] = {"total": 2 * total, "big": 2 * regional[1], "throat": 2 * regional[2], "small": 2 * regional[3]}

    level = MESH_LEVELS["fine"]
    isolated = {
        "big": build_mesh(disk_plc(GEOMETRY_UM["big_radius"], level)),
        "small": build_mesh(disk_plc(GEOMETRY_UM["small_radius"], level)),
        "throat": build_mesh(throat_plc(level)),
    }
    uncoupled = {}
    for name, mesh in isolated.items():
        start = time.perf_counter()
        total, _ = BlochTorreyStepper(mesh, TIME_STEP_MS).run(n_steps)
        results["timing_s"][f"isolated_{name}"] = time.perf_counter() - start
        results["meshes"][f"isolated_{name}"] = {"mesh": mesh, **mesh_summary(mesh)}
        uncoupled[name] = total
    # One period contains B, S and two throats.
    uncoupled["total"] = uncoupled["big"] + uncoupled["small"] + 2 * uncoupled["throat"]
    results["uncoupled"] = uncoupled
    results["analytic_t2_ms"] = {
        "big_disk_bessel": bessel_disk_t2_ms(GEOMETRY_UM["big_radius"]),
        "small_disk_bessel": bessel_disk_t2_ms(GEOMETRY_UM["small_radius"]),
        "throat_slab": slab_t2_ms(h),
    }
    results["isolated_fit_t2_ms"] = {name: fit_t2_ms(times, uncoupled[name], 0.9, 0.05) for name in ("big", "small", "throat")}
    # Effective T2 of each pore-body region inside the connected chain, fitted directly to the regional
    # magnetization (no inversion): early decay down to 10% of the region's initial magnetization.
    conn = results["connected_fine"]
    results["connected_region_fit_t2_ms"] = {name: fit_t2_ms(times, conn[name], 0.9, 0.1) for name in ("big", "small")}
    return results


def fit_t2_ms(times, signal, upper, lower):
    y = signal / signal[0]
    mask = (y < upper) & (y > lower)
    return float(-1.0 / np.polyfit(times[mask], np.log(y[mask]), 1)[0])


def invert(times, signal, name, rng, noise=NOISE_RELATIVE):
    normalized = signal / signal[0]
    noisy = normalized + rng.normal(0.0, noise, normalized.size)
    lcurve = invert_single_signal_lcurve(times, noisy, signal_name=name, config=LCurveConfig(**T2_BINS), alpha_values=LCURVE_ALPHAS)
    if not lcurve.used_range_filter or lcurve.best_index in (0, LCURVE_ALPHAS.size - 1):
        raise RuntimeError(f"{name}: L-curve criterion not met inside the alpha scan")
    fixed = invert_single_signal_nnls(times, noisy, signal_name=name, config=NnlsConfig(regularization=FIXED_ALPHA, **T2_BINS))
    return {"noisy": noisy, "noise": noise, "lcurve": lcurve, "fixed": fixed}


def run_inversions(nmr: dict) -> dict:
    times = nmr["times_ms"]
    out = {}
    # Same noise realization everywhere so differences are numerical or physical, not noise.
    for key in ("connected_fine", "connected_coarse"):
        out[key] = invert(times, nmr[key]["total"], key, np.random.default_rng(NOISE_SEED))
    out["uncoupled"] = invert(times, nmr["uncoupled"]["total"], "uncoupled", np.random.default_rng(NOISE_SEED))
    for name in ("big", "small", "throat"):
        out[f"isolated_{name}"] = invert(times, nmr["uncoupled"][name], f"isolated_{name}", np.random.default_rng(NOISE_SEED))
    out["connected_fine_noise1pct"] = invert(
        times, nmr["connected_fine"]["total"], "connected_fine_noise1pct", np.random.default_rng(NOISE_SEED), NOISE_SENSITIVITY
    )
    return out


# ---------------------------------------------------------------------------
# SIP package
# ---------------------------------------------------------------------------
def ensure_sip_results(sip_root: Path, sip_results: Path, out: Path) -> dict:
    manifest_path = sip_results / "provenance" / "run_manifest.json"
    if not manifest_path.exists():
        geometry_json = out / "source_data" / "shared_geometry_m.json"
        geometry_json.write_text(
            json.dumps({f"{k}_m": v * 1e-6 for k, v in GEOMETRY_UM.items()}, indent=2),
            encoding="utf-8",
        )
        script = sip_root / "code" / "scripts" / "sip_simulation" / "run_ideal_two_pore_channel_sip2d.py"
        subprocess.run([sys.executable, str(script), "--output-dir", str(sip_results), "--geometry-json", str(geometry_json)], check=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"SIP result package is not a complete run: {manifest.get('status')}")
    for key, value in GEOMETRY_UM.items():
        sip_value = manifest["geometry"][f"{key}_m"] * 1e6
        if not math.isclose(sip_value, value, rel_tol=1e-9):
            raise RuntimeError(f"SIP geometry {key}={sip_value} um differs from NMR geometry {value} um")
    return manifest


def load_sip(sip_results: Path, manifest: dict) -> dict:
    sweeps = {run["name"]: pd.read_csv(sip_results / "simulation_sweeps" / run["name"] / "sweep_results.csv") for run in manifest["runs"]}
    debye = {name: pd.read_csv(sip_results / "source_data" / f"debye_decomposition_{name}.csv") for name in sweeps}
    lam = pd.read_csv(sip_results / "source_data" / "dynamic_pore_size_table.csv")
    return {"sweeps": sweeps, "debye": debye, "lambda": lam, "region_map": np.load(sip_results / "source_data" / "region_map.npy"),
            "fields": np.load(sip_results / "source_data" / "dc_pore_only_fields.npz")}


# ---------------------------------------------------------------------------
# Actual size distribution (analytic local thickness)
# ---------------------------------------------------------------------------
def actual_size_histogram(lam: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Area-weighted local-thickness (maximal inscribed disk radius) histogram.

    For this geometry the local thickness is exact: every point of a disk has local thickness R,
    every throat point outside the disks has h. Areas are the analytic region areas.
    """

    edges = 10.0 ** (np.arange(-20.5, 21.0, 1.0) / 10.0)  # 10 bins/decade centred on 10^(k/10)
    rows = lam[lam["label"] > 0]
    total = rows["analytic_area_m2"].sum()
    body = np.zeros(edges.size - 1)
    throat = np.zeros(edges.size - 1)
    for _, row in rows.iterrows():
        k = np.searchsorted(edges, row["analytic_inscribed_radius_m"] * 1e6) - 1
        target = body if "body" in row["region"] else throat
        target[k] += row["analytic_area_m2"] / total
    frame = pd.DataFrame(
        {
            "bin_left_um": edges[:-1],
            "bin_right_um": edges[1:],
            "bin_center_um": np.sqrt(edges[:-1] * edges[1:]),
            "pore_body_area_fraction": body,
            "throat_area_fraction": throat,
            "pore_body_fraction_within_class": body / body.sum(),
            "throat_fraction_within_class": throat / throat.sum(),
        }
    )
    return frame, edges


def t2_extraction_table(nmr, inversions) -> pd.DataFrame:
    """Back-calculated sizes from four T2 extraction routes.

    direct_fit: single-exponential fit of the isolated pore's own decay (no inversion);
    isolated_nnls: L-curve NNLS of that pore's decay alone; composite_nnls: L-curve NNLS of the area-weighted
    sum of the isolated decays (B + S + 2 throats); connected_nnls: L-curve NNLS of the connected chain.
    Pore bodies use r = 2 rho2 / (1/T2 - 1/T2B); the slit throat uses h = rho2 / (1/T2 - 1/T2B).
    """

    pores = {"big": (GEOMETRY_UM["big_radius"], 2.0, "big_disk_bessel"),
             "small": (GEOMETRY_UM["small_radius"], 2.0, "small_disk_bessel"),
             "throat": (0.5 * GEOMETRY_UM["throat_width"], 1.0, "throat_slab")}
    size = lambda t2, factor: factor * RHO2_UM_MS / (1.0 / t2 - 1.0 / BULK_T2_MS)
    rows = []
    for pore, (true_size, factor, analytic_key) in pores.items():
        target = nmr["analytic_t2_ms"][analytic_key]
        fit = nmr["isolated_fit_t2_ms"][pore]
        routes = [("direct_fit", fit, fit)]
        for route, key in (("isolated_nnls", f"isolated_{pore}"), ("composite_nnls", "uncoupled"), ("connected_nnls", "connected_fine")):
            inv = inversions[key]["lcurve"]
            modes = spectrum_peaks(inv.t2_bins_ms, inv.spectrum)
            if route == "isolated_nnls":
                mode = max(modes, key=lambda m: m["fraction"])
            else:
                # the mode closest to this pore's isolated T2; unresolved if none within a factor of 2
                mode = min(modes, key=lambda m: abs(np.log(m["position"] / target)))
                if abs(np.log(mode["position"] / target)) > np.log(2.0):
                    mode = None
            routes.append((route, mode["position"] if mode else np.nan, mode["t2lm"] if mode else np.nan))
        for route, t2_peak, t2lm in routes:
            rows.append({
                "pore": pore, "true_size_um": true_size, "conversion": "disk 2*rho2" if factor == 2 else "slit rho2",
                "route": route, "t2_peak_ms": t2_peak, "size_peak_um": size(t2_peak, factor),
                "error_peak_pct": 100 * (size(t2_peak, factor) / true_size - 1),
                "t2lm_ms": t2lm, "size_t2lm_um": size(t2lm, factor),
                "error_t2lm_pct": 100 * (size(t2lm, factor) / true_size - 1),
            })
    return pd.DataFrame(rows)


def density_in_log_radius(t2_bins, amplitude):
    """Re-express a T2-bin amplitude as a density per log10(r_NMR) (the bulk term makes the map non-affine)."""

    r = nmr_radius_um(t2_bins)
    valid = np.isfinite(r) & (r > 0)
    log_r = np.log10(r[valid])
    width = np.gradient(log_r)
    return r[valid], amplitude[valid] / width


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig, out: Path, name: str) -> list[str]:
    paths = []
    for ext, dpi in (("png", 600), ("pdf", None)):
        path = out / "figures" / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def panel_label(ax, tag):
    ax.text(-0.02, 1.02, tag, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom", color=INK)


def draw_mesh(ax, mesh, linewidth):
    xy = np.array([[n.pos().x(), n.pos().y()] for n in mesh.nodes()])
    tri = np.array([[c.node(i).id() for i in range(3)] for c in mesh.cells()])
    ax.triplot(xy[:, 0], xy[:, 1], tri, color=INK2, lw=linewidth)


def figure_geometry(nmr, sip, out):
    g = GEOMETRY_UM
    rb, rs, w, length, margin = g["big_radius"], g["small_radius"], g["throat_width"], g["throat_length"], g["solid_margin"]
    period, height = 2 * (rb + rs + length), 2 * (rb + margin)
    xb, xs = 0.5 * length + rb, 1.5 * length + 2 * rb + rs
    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.32)
    ax = fig.add_subplot(grid[0, :])
    ax.add_patch(Rectangle((0, -height / 2), period, height, facecolor="#e7e3d8", edgecolor=INK2, lw=0.8))
    # The 0.1-um throat is invisible at this scale; draw it as a line (width not to scale).
    ax.plot([0, period], [0, 0], color="#2c6485", lw=1.2, solid_capstyle="butt")
    ax.add_patch(Circle((xb, 0), rb, facecolor="#8ec6e8", edgecolor="#2c6485", lw=0.8))
    ax.add_patch(Circle((xs, 0), rs, facecolor="#8ec6e8", edgecolor="#2c6485", lw=0.8))
    ax.add_patch(Rectangle((xb, -rb - 0.6), xs - xb, 2 * rb + 1.2, fill=False, edgecolor=C_NMR, ls="--", lw=1.0))
    ax.annotate("", (xb, 0), (xb - rb * 0.7071, rb * 0.7071), arrowprops=dict(arrowstyle="<-", lw=0.6, color=INK))
    ax.text(xb, -6.0, f"R_B = {rb:g} um", ha="center", va="center", fontsize=6.5, color=INK)
    ax.annotate("", (xs, 0), (xs, rs), arrowprops=dict(arrowstyle="<-", lw=0.6, color=INK))
    ax.text(xs, rs + 1.6, f"R_S = {rs:g} um", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.text(0.5 * (xb + rb + xs - rs), -1.5, f"throat\nL = {length:g} um\nw = {w:g} um*", ha="center", va="top", fontsize=6, color=INK)
    ax.text(0, -height / 2 - 1.2, "Dashed box: NMR half cell (mirror planes through the pore centres, no flux).\n"
            "SIP: field E0 along x, periodic cell.  *Throat line width not to scale.", ha="left", va="top", fontsize=6.2, color=INK)
    ax.set(xlim=(-1, period + 1), ylim=(-height / 2 - 6.5, height / 2 + 1), aspect="equal", xlabel="x (um)", ylabel="y (um)")
    ax.set_title("Periodic unit cell: big pore B - slit throat - small pore S - slit throat (2-D, per unit depth)", loc="left", fontsize=7.5)
    panel_label(ax, "a")

    mesh = nmr["meshes"]["connected_fine"]["mesh"]
    ax = fig.add_subplot(grid[1, 0])
    draw_mesh(ax, mesh, 0.08)
    ax.set(xlim=(-0.5, rb + length + rs + 0.5), ylim=(-rb - 0.5, rb + 0.5), aspect="equal", xlabel="x (um)", ylabel="y (um)")
    ax.set_title(f"pyGIMLi mesh, NMR ({nmr['meshes']['connected_fine']['nodes']} nodes)", loc="left", fontsize=7)
    panel_label(ax, "b")
    ax = fig.add_subplot(grid[1, 1])
    draw_mesh(ax, mesh, 0.2)
    ax.set(xlim=(rb - 0.4, rb + 0.6), ylim=(-0.35, 0.35), aspect="equal", xlabel="x (um)")
    ax.set_title("NMR mesh at the throat mouth", loc="left", fontsize=7)
    panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 2])
    fields = sip["fields"]
    xe, ye = fields["x_edges_m"] * 1e6, fields["y_edges_m"] * 1e6 - height / 2
    colors = matplotlib.colors.ListedColormap(["#e7e3d8", C_BODY, C_BODY, C_THROAT, C_THROAT])
    ax.pcolormesh(xe, ye, sip["region_map"], cmap=colors, vmin=-0.5, vmax=4.5, edgecolors="white", linewidth=0.05, rasterized=True)
    x_mouth = 0.5 * length + 2 * rb
    ax.set(xlim=(x_mouth - 0.4, x_mouth + 0.6), ylim=(-0.35, 0.35), aspect="equal", xlabel="x (um)")
    ax.set_title("SIP finite-volume grid at the mouth", loc="left", fontsize=7)
    panel_label(ax, "d")
    return save(fig, out, "fig1_geometry_meshes")


def figure_size_selection(nmr, out):
    g = GEOMETRY_UM
    h = 0.5 * g["throat_width"]
    radius = np.logspace(-2.3, 2, 300)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

    ax = axes[0]
    surface_only = radius / (2 * RHO2_UM_MS)
    ax.loglog(radius, surface_only, color=INK2, ls="--", lw=1.0, label="surface only, R/(2 rho2)")
    ax.loglog(radius, nmr_t2_from_radius_ms(radius), color=C_NMR, label="disk + bulk T2B = 3 s")
    ax.loglog(radius, 1 / (RHO2_UM_MS / radius + 1 / BULK_T2_MS), color=C_THROAT, lw=1.0, label="slit, half-width h")
    ax.axhspan(1e-4, 3 * TIME_STEP_MS, color=GRID, lw=0)
    ax.axhline(BULK_T2_MS, color=INK2, lw=0.6, ls=":")
    ax.text(0.012, BULK_T2_MS * 1.3, "T2B", fontsize=6, color=INK2)
    ax.text(0.012, 0.25, "< 3 TE", fontsize=6, color=INK2)
    for r, color in ((g["big_radius"], C_BODY), (g["small_radius"], C_BODY)):
        ax.plot(r, nmr_t2_from_radius_ms(r), "o", ms=5, color=color, mec="white", mew=0.8)
    ax.plot(h, 1 / (RHO2_UM_MS / h + 1 / BULK_T2_MS), "o", ms=5, color=C_THROAT, mec="white", mew=0.8)
    ax.set(xlabel="Pore size r or h (um)", ylabel="T2 (ms)", ylim=(0.1, 3e4))
    ax.legend(fontsize=5.8, loc="lower right")
    ax.set_title("NMR window", loc="left", fontsize=7.5)
    panel_label(ax, "a")

    ax = axes[1]
    ax.loglog(radius, sip_frequency_from_radius_hz(radius), color=C_SIP, label="Stern: f = D+/(pi r^2), D+ = 3.8e-12")
    ax.loglog(radius, sip_frequency_from_radius_hz(radius, 1.3e-9), color=INK2, ls="--", lw=1.0, label="same with D = 1.3e-9 (Niu 2020)")
    length_axis = np.logspace(0, 2.3, 100)
    ax.loglog(length_axis, 2 * 1.3e-9 / (np.pi * (length_axis * 1e-6) ** 2), color=C_THROAT, lw=1.0, ls="-.", label="membrane: f = 2D/(pi L^2) vs L")
    ax.axhspan(1e-4, 1e5, color=GRID, lw=0, zorder=0)
    for r, color in ((g["big_radius"], C_BODY), (g["small_radius"], C_BODY), (h, C_THROAT)):
        ax.plot(r, sip_frequency_from_radius_hz(r), "o", ms=5, color=color, mec="white", mew=0.8)
    ax.plot(g["throat_length"], 2 * 1.3e-9 / (np.pi * (g["throat_length"] * 1e-6) ** 2), "s", ms=5, color=C_THROAT, mec="white", mew=0.8)
    ax.set(xlabel="r, h or throat length L (um)", ylabel="Relaxation frequency (Hz)", ylim=(1e-6, 1e8))
    ax.text(0.006, 2e-4, "simulated band 1e-4 - 1e5 Hz", fontsize=6, color=INK2)
    ax.legend(fontsize=5.8, loc="upper right")
    ax.set_title("SIP window", loc="left", fontsize=7.5)
    panel_label(ax, "b")

    ax = axes[2]
    half_widths = np.logspace(-3, 0, 100)
    rs = g["small_radius"]
    sink = 2 * 2 * np.sqrt(DIFFUSION_UM2_MS * RHO2_UM_MS * half_widths) / (np.pi * rs**2)
    coupled = 1 / (2 * RHO2_UM_MS / rs + sink + 1 / BULK_T2_MS)
    ax.semilogx(half_widths, coupled / bessel_disk_t2_ms(rs), color=C_NMR, label="small-pore T2 (connected / isolated), sink model")
    ratio = nmr["connected_region_fit_t2_ms"]["small"] / bessel_disk_t2_ms(rs)
    ax.plot(h, ratio, "D", ms=5, color=C_NMR, mec=INK, mew=0.6, label="this simulation (small-pore region fit)")
    ax.axvline(h, color=C_THROAT, lw=0.8, ls=":")
    ax.set(xlabel="Throat half-width h (um)", ylabel="T2 ratio", ylim=(0, 1.05))
    ax.legend(fontsize=5.8, loc="lower left")
    ax.set_title("Throat-wall sink on the small pore", loc="left", fontsize=7.5)
    panel_label(ax, "c")
    for a in axes:
        a.grid(color=GRID, lw=0.5, which="major")
    return save(fig, out, "fig2_size_selection_windows")


def figure_nmr(nmr, inversions, out):
    times = nmr["times_ms"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    conn = nmr["connected_fine"]
    m0 = conn["total"][0]
    ax.semilogy(times, conn["total"] / m0, color=C_NMR, label="connected chain, total")
    ax.semilogy(times, conn["big"] / m0, color=C_BODY, lw=1.0, label="  in big pore")
    ax.semilogy(times, conn["small"] / m0, color=C_BODY, lw=1.0, ls="--", label="  in small pore")
    ax.semilogy(times, conn["throat"] / m0, color=C_THROAT, lw=1.0, label="  in throats")
    unc = nmr["uncoupled"]
    ax.semilogy(times, unc["total"] / unc["total"][0], color=INK2, lw=1.0, ls=":", label="uncoupled (isolated B + S + throats)")
    ax.set(xlabel="Time (ms)", ylabel="M(t) / M(0)", ylim=(1e-7, 1.5))
    ax.legend(fontsize=6, loc="lower left")
    panel_label(ax, "a")

    ax = axes[0, 1]
    lc, fx, un = inversions["connected_fine"]["lcurve"], inversions["connected_fine"]["fixed"], inversions["uncoupled"]["lcurve"]
    noisy = inversions["connected_fine_noise1pct"]["lcurve"]
    ax.semilogx(lc.t2_bins_ms, lc.spectrum, color=C_NMR, label=f"connected, L-curve NNLS (alpha = {lc.best_regularization:.3g})")
    ax.fill_between(lc.t2_bins_ms, lc.spectrum, color=C_NMR, alpha=0.18, lw=0)
    ax.semilogx(fx.t2_bins_ms, fx.spectrum, color=C_GREEN, lw=1.0, ls="--", label=f"connected, fixed NNLS (alpha = {FIXED_ALPHA:g})")
    ax.semilogx(un.t2_bins_ms, un.spectrum, color=INK2, lw=1.0, ls=":", label="uncoupled reference, L-curve NNLS")
    ax.semilogx(noisy.t2_bins_ms, noisy.spectrum, color=C_VIOLET, lw=1.0, ls="-.", label=f"connected, 1% noise, L-curve (alpha = {noisy.best_regularization:.3g})")
    analytic = nmr["analytic_t2_ms"]
    for key, label in (("big_disk_bessel", "B"), ("small_disk_bessel", "S"), ("throat_slab", "throat")):
        ax.axvline(analytic[key], color=C_THROAT if key == "throat_slab" else C_BODY, lw=0.7, ls=":")
        ax.text(analytic[key], 0.92, f" {label}", transform=ax.get_xaxis_transform(), fontsize=6, color=INK2, va="top")
    ax.set(xlabel="T2 (ms)", ylabel="Amplitude per bin", xlim=(1, 1e4))
    ax.legend(fontsize=5.8, loc="center left", title=f"SNR = {1/NOISE_RELATIVE:g} unless stated", title_fontsize=5.8)
    ax.text(0.01, 0.99, "dotted: isolated-pore analytic T2", transform=ax.transAxes, fontsize=5.8, color=INK2, va="top")
    panel_label(ax, "b")

    ax = axes[1, 0]
    i = lc.best_index
    ax.loglog(lc.residual_norms, lc.roughness_norms, "o-", ms=2.5, lw=0.8, color=C_NMR)
    ax.loglog(lc.residual_norms[i], lc.roughness_norms[i], "o", ms=8, mfc="none", mec=INK, mew=1.0)
    ax.annotate(f"selected alpha = {lc.best_regularization:.3g}\n(alpha^2 eta / zeta = {lc.slope_reciprocal_values[i]:.2f}, target 0.25)",
                (lc.residual_norms[i], lc.roughness_norms[i]), (14, 6), textcoords="offset points", fontsize=6.5, color=INK)
    ax.set(xlabel="Weighted residual norm", ylabel="Roughness norm")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("L-curve, nmr_t2 (connected chain, fine mesh)", loc="left", fontsize=7)
    panel_label(ax, "c")

    ax = axes[1, 1]
    co = inversions["connected_coarse"]["lcurve"]
    ax.semilogx(lc.t2_bins_ms, lc.spectrum, color=C_NMR, label=f"fine mesh ({nmr['meshes']['connected_fine']['nodes']} nodes)")
    ax.semilogx(co.t2_bins_ms, co.spectrum, color=C_VIOLET, lw=1.0, ls="--", label=f"coarse mesh ({nmr['meshes']['connected_coarse']['nodes']} nodes)")
    diff = np.max(np.abs(nmr["connected_fine"]["total"] / nmr["connected_fine"]["total"][0] - nmr["connected_coarse"]["total"] / nmr["connected_coarse"]["total"][0]))
    ax.text(0.02, 0.95, f"max |dM/M0| between meshes = {diff:.1e}", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
    ax.set(xlabel="T2 (ms)", ylabel="Amplitude per bin", xlim=(1, 1e4))
    ax.legend(fontsize=6, loc="center left")
    ax.set_title("Mesh convergence (same noise realization)", loc="left", fontsize=7)
    panel_label(ax, "d")
    for a in axes.ravel():
        a.grid(color=GRID, lw=0.5)
    return save(fig, out, "fig3_nmr_decay_t2_inversion"), float(diff)


def figure_sip(sip, out):
    sweeps, debye = sip["sweeps"], sip["debye"]
    mech = [("interfacial", C_BODY, "-", "interfacial (Maxwell-Wagner)"), ("pore_regional", C_THROAT, "-", "pore / Stern, region-resolved"),
            ("pore_uniform_niu", C_THROAT, "--", "pore / Stern, Niu uniform"), ("membrane", C_NMR, "-", "membrane (Titov throat)"),
            ("all_regional", C_SIP, "-", "all, region-resolved"), ("all_uniform_niu", C_SIP, "--", "all, Niu uniform")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for name, color, ls, label in mech:
        frame = sweeps[name]
        f = frame["frequency_hz"]
        axes[0].loglog(f, frame["sigma_real_s_m"], color=color, ls=ls, lw=1.2, label=label)
        axes[1].loglog(f, frame["sigma_imag_s_m"].where(frame["sigma_imag_s_m"] > 0), color=color, ls=ls, lw=1.2, label=label)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="sigma' (S/m)")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="sigma'' (S/m)", ylim=(1e-12, None))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    for name, color, ls, label in mech[1:]:
        table = debye[name]
        amp = table["amplitude_s_m"]
        axes[2].semilogx(table["radius_um"], amp / amp.max(), color=color, ls=ls, lw=1.2, label=label)
    for r, color in ((GEOMETRY_UM["big_radius"], C_BODY), (GEOMETRY_UM["small_radius"], C_BODY), (0.5 * GEOMETRY_UM["throat_width"], C_THROAT)):
        axes[2].axvline(r, color=color, lw=0.7, ls=":")
    axes[2].set(xlabel="r = sqrt(2 tau D+) (um)", ylabel="Debye amplitude / max", xlim=(3e-3, 60), ylim=(0, 1.05))
    for a, tag in zip(axes, "abc"):
        a.grid(color=GRID, lw=0.5)
        panel_label(a, tag)
    return save(fig, out, "fig4_sip_mechanisms_debye")


def overlay_series(hist, inversions, sip):
    lc = inversions["connected_fine"]["lcurve"]
    r_nmr, dens = density_in_log_radius(lc.t2_bins_ms, lc.spectrum)
    un = inversions["uncoupled"]["lcurve"]
    r_unc, dens_unc = density_in_log_radius(un.t2_bins_ms, un.spectrum)
    dd = sip["debye"]["all_regional"]
    dd_uni = sip["debye"]["all_uniform_niu"]
    sweep = sip["sweeps"]["all_regional"]
    sweep_uni = sip["sweeps"]["all_uniform_niu"]
    return {
        "nmr_r": r_nmr, "nmr": dens / dens.max(),
        "nmr_unc_r": r_unc, "nmr_unc": dens_unc / dens_unc.max(),
        "dd_r": dd["radius_um"].to_numpy(), "dd": (dd["amplitude_s_m"] / dd["amplitude_s_m"].max()).to_numpy(),
        "dd_uni_r": dd_uni["radius_um"].to_numpy(), "dd_uni": (dd_uni["amplitude_s_m"] / dd_uni["amplitude_s_m"].max()).to_numpy(),
        "imag_r": sip_radius_from_frequency_um(sweep["frequency_hz"]), "imag": (sweep["sigma_imag_s_m"] / sweep["sigma_imag_s_m"].max()).to_numpy(),
        "imag_uni_r": sip_radius_from_frequency_um(sweep_uni["frequency_hz"]), "imag_uni": (sweep_uni["sigma_imag_s_m"] / sweep_uni["sigma_imag_s_m"].max()).to_numpy(),
    }


def add_conversion_axes(ax):
    top = ax.secondary_xaxis("top", functions=(lambda r: nmr_t2_from_radius_ms(np.clip(r, 1e-9, None)), lambda t: nmr_radius_um(np.clip(t, 1e-9, BULK_T2_MS * 0.999999))))
    top.set_xlabel("NMR T2 (ms)  [r = 2 rho2 / (1/T2 - 1/T2B)]", color=INK, fontsize=7)
    top.tick_params(colors=INK2, labelsize=6.5)
    top2 = ax.secondary_xaxis(1.22, functions=(lambda r: sip_frequency_from_radius_hz(np.clip(r, 1e-9, None)), lambda f: sip_radius_from_frequency_um(np.clip(f, 1e-30, None))))
    top2.set_xlabel("SIP relaxation frequency (Hz)  [r = sqrt(2 tau D+), tau = 1/(2 pi f)]", color=INK, fontsize=7)
    top2.tick_params(colors=INK2, labelsize=6.5)
    return top, top2


def figure_overlay(hist, edges, series, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    widths = edges[1:] - edges[:-1]
    body = hist["pore_body_area_fraction"].to_numpy()
    throat = hist["throat_area_fraction"].to_numpy()
    ax.bar(edges[:-1], body / body.max(), width=widths, align="edge", color=C_BODY, alpha=0.55, edgecolor="white", lw=0.8,
           label="Actual pore bodies: area fraction / class max")
    ax.bar(edges[:-1], throat / throat.max(), width=widths, align="edge", color=C_THROAT, alpha=0.65, edgecolor="white", lw=0.8,
           label="Actual throat: area fraction / class max")
    for k in np.flatnonzero(body + throat):
        value = body[k] + throat[k]
        height = body[k] / body.max() if body[k] else throat[k] / throat.max()
        # Keep short bars' labels above the neighbouring NMR peak.
        ax.text(hist["bin_center_um"].iloc[k], max(height, 0.14) + 0.02, f"{100*value:.2g}% of\npore area", ha="center", va="bottom", fontsize=6, color=INK)
    ax.plot(series["nmr_r"], series["nmr"], color=C_NMR, lw=1.8, label="NMR T2 (connected chain, L-curve NNLS), density / max")
    ax.fill_between(series["nmr_r"], series["nmr"], color=C_NMR, alpha=0.15, lw=0)
    ax.plot(series["dd_r"], series["dd"], color=C_SIP, lw=1.8, label="SIP Debye decomposition (all mechanisms), amplitude / max")
    ax.plot(series["imag_r"], series["imag"], color=C_SIP, lw=1.2, ls="--", label="SIP sigma''(f) at r = sqrt(D+/(pi f)), / max")
    ax.plot(series["dd_uni_r"], series["dd_uni"], color=C_VIOLET, lw=1.0, ls=":", label="SIP Debye decomposition, Niu uniform upscaling (comparison)")
    ax.set_xscale("log")
    ax.set(xlim=(5e-3, 60), ylim=(0, 1.12), xlabel="Pore radius r (um)", ylabel="Normalized amplitude (each series / its maximum)")
    ax.grid(color=GRID, lw=0.5, which="major")
    add_conversion_axes(ax)
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, columnspacing=1.5)
    fig.text(0.5, -0.17,
             "2-D periodic chain (per unit depth). NMR: rho2 = 5 um/s, D0 = 2e-9 m2/s, T2B = 3 s, SNR 1000, circular-pore shape factor. "
             "SIP: region-resolved Niu (2020) mechanisms; D+ = 3.8e-12 m2/s (Stern), D = 1.3e-9 m2/s (membrane).",
             ha="center", va="top", fontsize=5.8, color=INK2, wrap=True)
    return save(fig, out, "fig5_overlay_actual_sizes_nmr_t2_sip")


def figure_overlay_stacked(hist, edges, series, inversions, nmr, out):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"hspace": 0.12})
    widths = edges[1:] - edges[:-1]
    ax = axes[0]
    ax.bar(edges[:-1], 100 * hist["pore_body_area_fraction"], width=widths, align="edge", color=C_BODY, edgecolor="white", lw=0.8, label="pore bodies")
    ax.bar(edges[:-1], 100 * hist["throat_area_fraction"], width=widths, align="edge", color=C_THROAT, edgecolor="white", lw=0.8, label="throats")
    ax.set(yscale="log", ylim=(0.05, 200), ylabel="Pore-area fraction (%)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("Actual sizes (local thickness = inscribed radius, area weighted)", loc="left", fontsize=7.5)
    add_conversion_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    lc = inversions["connected_fine"]["lcurve"]
    r, dens = density_in_log_radius(lc.t2_bins_ms, lc.spectrum)
    ax.plot(r, dens, color=C_NMR, label="connected chain, L-curve NNLS")
    ax.fill_between(r, dens, color=C_NMR, alpha=0.15, lw=0)
    fx = inversions["connected_fine"]["fixed"]
    r, dens = density_in_log_radius(fx.t2_bins_ms, fx.spectrum)
    ax.plot(r, dens, color=C_GREEN, lw=1.0, ls="--", label=f"connected chain, fixed NNLS (alpha = {FIXED_ALPHA:g})")
    un = inversions["uncoupled"]["lcurve"]
    r, dens = density_in_log_radius(un.t2_bins_ms, un.spectrum)
    ax.plot(r, dens, color=INK2, lw=1.0, ls=":", label="uncoupled reference, L-curve NNLS")
    for key, rr in (("big_disk_bessel", GEOMETRY_UM["big_radius"]), ("small_disk_bessel", GEOMETRY_UM["small_radius"])):
        ax.axvline(nmr_radius_um(nmr["analytic_t2_ms"][key]), color=C_BODY, lw=0.7, ls=":")
    ax.set(ylabel="Amplitude per log10 r")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("NMR T2 distribution mapped with r = 2 rho2 / (1/T2 - 1/T2B)", loc="left", fontsize=7.5)
    panel_label(ax, "b")

    ax = axes[2]
    ax.plot(series["dd_r"], series["dd"], color=C_SIP, label="Debye decomposition, region-resolved (all)")
    ax.plot(series["imag_r"], series["imag"], color=C_SIP, lw=1.0, ls="--", label="sigma'' / max, region-resolved (all)")
    ax.plot(series["dd_uni_r"], series["dd_uni"], color=C_VIOLET, lw=1.2, label="Debye decomposition, Niu uniform (all)")
    ax.plot(series["imag_uni_r"], series["imag_uni"], color=C_VIOLET, lw=1.0, ls="--", label="sigma'' / max, Niu uniform (all)")
    ax.set(xscale="log", xlim=(5e-3, 60), ylim=(0, 1.08), xlabel="Pore radius r (um)", ylabel="Normalized amplitude")
    ax.legend(fontsize=6.5, loc="upper center")
    ax.set_title("SIP mapped with r = sqrt(2 tau D+), D+ = 3.8e-12 m2/s", loc="left", fontsize=7.5)
    panel_label(ax, "c")
    for a in axes:
        a.grid(color=GRID, lw=0.5, which="major")
        for rr, color in ((GEOMETRY_UM["big_radius"], C_BODY), (GEOMETRY_UM["small_radius"], C_BODY), (0.5 * GEOMETRY_UM["throat_width"], C_THROAT)):
            a.axvline(rr, color=color, lw=0.6, ls="-", alpha=0.35, zorder=0)
    return save(fig, out, "fig6_stacked_actual_nmr_sip_absolute")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "simulation_outputs" / "ideal_two_pore_channel_nmr_sip_20261012")
    parser.add_argument("--sip-root", type=Path, default=ROOT.parent / "SIP模拟" / "sip模拟")
    parser.add_argument("--sip-results", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    sip_results = (args.sip_results or args.sip_root / "results" / "ideal_two_pore_channel_sip2d_20261012").resolve()
    for sub in ("figures", "source_data"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    style()
    t0 = time.perf_counter()

    sip_manifest = ensure_sip_results(args.sip_root, sip_results, out)
    sip = load_sip(sip_results, sip_manifest)
    print("NMR simulation ...", flush=True)
    nmr = simulate_nmr(out)
    print(f"  stepper vs pg.solve max rel diff = {nmr['verification']['max_rel_diff_vs_pg_solve_3_steps']:.2e}", flush=True)
    print("NMR inversion ...", flush=True)
    inversions = run_inversions(nmr)

    hist, edges = actual_size_histogram(sip["lambda"])
    series = overlay_series(hist, inversions, sip)

    # Source data.
    times = nmr["times_ms"]
    decay = pd.DataFrame({"time_ms": times})
    for key in ("connected_fine", "connected_coarse"):
        for part, values in nmr[key].items():
            decay[f"{key}_{part}"] = values / nmr[key]["total"][0]
    for part, values in nmr["uncoupled"].items():
        decay[f"uncoupled_{part}"] = values / nmr["uncoupled"]["total"][0]
    decay["connected_fine_total_noisy_input"] = inversions["connected_fine"]["noisy"]
    decay.iloc[::4].to_csv(out / "source_data" / "nmr_decays_every_1ms.csv", index=False)
    spectra = pd.DataFrame({"t2_ms": inversions["connected_fine"]["lcurve"].t2_bins_ms, "radius_nmr_um": nmr_radius_um(inversions["connected_fine"]["lcurve"].t2_bins_ms)})
    for key, inv in inversions.items():
        spectra[f"{key}_lcurve"] = inv["lcurve"].spectrum
        spectra[f"{key}_fixed_alpha{FIXED_ALPHA:g}"] = inv["fixed"].spectrum
    spectra.to_csv(out / "source_data" / "nmr_t2_spectra.csv", index=False)
    t2_extraction_table(nmr, inversions).to_csv(out / "source_data" / "t2_extraction_and_size_backcalculation.csv", index=False)
    lc = inversions["connected_fine"]["lcurve"]
    pd.DataFrame({"alpha": lc.alpha_values, "residual_norm": lc.residual_norms, "roughness_norm": lc.roughness_norms,
                  "reciprocal_slope": lc.slope_reciprocal_values, "selected": np.arange(lc.alpha_values.size) == lc.best_index}).to_csv(
        out / "source_data" / "nmr_lcurve_connected_fine.csv", index=False)
    hist.to_csv(out / "source_data" / "actual_pore_body_throat_size_histogram.csv", index=False)
    pd.concat(
        [pd.DataFrame({"series": name, "radius_um": series[f"{name}_r"], "normalized_amplitude": series[name]})
         for name in ("nmr", "nmr_unc", "dd", "dd_uni", "imag", "imag_uni")],
        ignore_index=True,
    ).to_csv(out / "source_data" / "overlay_series_normalized.csv", index=False)

    print("Figures ...", flush=True)
    figures = {}
    figures["geometry"] = figure_geometry(nmr, sip, out)
    figures["size_selection"] = figure_size_selection(nmr, out)
    figures["nmr"], mesh_decay_diff = figure_nmr(nmr, inversions, out)
    figures["sip"] = figure_sip(sip, out)
    figures["overlay"] = figure_overlay(hist, edges, series, out)
    figures["overlay_stacked"] = figure_overlay_stacked(hist, edges, series, inversions, nmr, out)

    def peaks_t2(inv):
        return [{"t2_peak_ms": p["position"], "radius_peak_um": float(nmr_radius_um(p["position"])),
                 "t2lm_ms": p["t2lm"], "radius_t2lm_um": float(nmr_radius_um(p["t2lm"])), "fraction": p["fraction"]}
                for p in spectrum_peaks(inv.t2_bins_ms, inv.spectrum)]

    manifest = {
        "status": "complete",
        "model_dimension": "2D (per unit depth). NMR on the mirror-symmetric half cell of the periodic chain; SIP on the full periodic cell.",
        "geometry_um": GEOMETRY_UM,
        "nmr": {
            "solver": "pyGIMLi P1 triangle mesh; implicit Euler Robin Bloch-Torrey identical to pg.solve(a=D dt, b=-(1+dt/T2B), f=u, Robin rho dt); cached LU",
            "diffusion_um2_ms": DIFFUSION_UM2_MS, "surface_relaxivity_um_ms": RHO2_UM_MS, "bulk_t2_ms": BULK_T2_MS,
            "time_step_ms": TIME_STEP_MS, "time_max_ms": TIME_MAX_MS,
            "parameter_source": "Auto_NMR.py defaults (Zhou & Zhang 2025, Table A1; rho2 = 5 um/s)",
            "boundary_conditions": "Robin rho2 on all solid walls (marker 1); no-flux on symmetry cuts through the pore centres (marker 2)",
            "mesh_levels": MESH_LEVELS,
            "meshes": {k: {kk: vv for kk, vv in v.items() if kk != "mesh"} for k, v in nmr["meshes"].items()},
            "stepper_vs_pg_solve_max_rel_diff_3_steps": nmr["verification"]["max_rel_diff_vs_pg_solve_3_steps"],
            "mesh_convergence_max_abs_decay_diff": mesh_decay_diff,
            "analytic_isolated_t2_ms": nmr["analytic_t2_ms"],
            "isolated_mono_exponential_fit_t2_ms": nmr["isolated_fit_t2_ms"],
            "connected_region_fit_t2_ms": nmr["connected_region_fit_t2_ms"],
            "timing_s": nmr["timing_s"],
        },
        "inversion": {
            "noise": {"relative_sigma": NOISE_RELATIVE, "sensitivity_relative_sigma": NOISE_SENSITIVITY, "seed": NOISE_SEED,
                      "note": "same Gaussian realization (scaled) for all inverted decays; SNR 1000 = stacked acquisition design choice"},
            "t2_bins": T2_BINS,
            "fixed_nnls_alpha": FIXED_ALPHA,
            "solver": "nmr_t2.nnls.solve_t2_nnls_matlab_compatible (via invert_single_signal_nnls / invert_single_signal_lcurve)",
            "lcurve_scan": {"alpha_min": float(LCURVE_ALPHAS.min()), "alpha_max": float(LCURVE_ALPHAS.max()), "count": int(LCURVE_ALPHAS.size)},
            "lcurve_selection": "nmr_t2 reciprocal-slope criterion alpha^2 eta / zeta -> 0.25 inside (0.1, 10); interior point of the scan required",
            "selected_alpha": {k: float(v["lcurve"].best_regularization) for k, v in inversions.items()},
            "peaks": {k: {"lcurve": peaks_t2(v["lcurve"]), "fixed": peaks_t2(v["fixed"])} for k, v in inversions.items()},
        },
        "sip": {
            "results_dir": str(sip_results),
            "status": sip_manifest["status"],
            "global_dynamic_pore_size_m": sip_manifest["global_dynamic_pore_size_m"],
            "dc_formation_factor": sip_manifest["dc_formation_factor"],
            "regions": sip_manifest["regions"],
            "debye_summaries": sip_manifest["debye"]["summaries"],
            "parameters": sip_manifest["parameters"],
        },
        "conversions": {
            "nmr": "r = 2 rho2 / (1/T2 - 1/T2B) (disk, fast diffusion); slit half-width would be h = rho2 / (1/T2 - 1/T2B)",
            "sip": "r = sqrt(2 tau D+), tau = 1/(2 pi f0), D+ = 3.8e-12 m2/s (user-specified)",
            "display": "physical conversions of each method's own axis; no display-only rescaling or shifting",
        },
        "outputs": {"figures": figures, "source_data": sorted(str(p) for p in (out / "source_data").glob("*"))},
        "provenance": {
            "python": sys.version,
            "pygimli": pg.__version__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "nmr_repo_commit": git_commit(ROOT),
            "sip_repo_commit": git_commit(args.sip_root),
            "runtime_s": time.perf_counter() - t0,
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(out), "runtime_s": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    main()
