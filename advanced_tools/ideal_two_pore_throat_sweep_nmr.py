"""Throat width / length sweep for the ideal 2-D big pore - throat - small pore NMR geometry.

Full (non-symmetry-reduced) closed geometry: one big disk B, one straight throat of width w and
length L (wall to wall), one small disk S. Robin rho2 on every wall. Same pyGIMLi mesh generator,
Bloch-Torrey stepper (verified against pg.solve) and nmr_t2 L-curve NNLS as
``ideal_two_pore_channel_nmr_sip.py``.

    python advanced_tools/ideal_two_pore_throat_sweep_nmr.py --check-symmetry   # question 1 check
    python advanced_tools/ideal_two_pore_throat_sweep_nmr.py                    # sweep + figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import advanced_tools.ideal_two_pore_channel_nmr_sip as base  # noqa: E402

BIG_RADIUS_UM = base.GEOMETRY_UM["big_radius"]
SMALL_RADIUS_UM = base.GEOMETRY_UM["small_radius"]
WIDTHS_UM = [0.1, 0.5, 1.0, 2.0, 4.0]
LENGTHS_UM = [2.0, 5.0, 15.0, 30.0]
BODY_AREA_UM2 = 0.06  # coarse/fine meshes of the base study agreed to 1e-4 in M(t)/M0
SEGMENT_UM = 0.15


def closed_plc(width, length):
    """Full B disk + throat + full S disk; region markers 1 (B), 2 (throat), 3 (S)."""

    import pygimli.meshtools as mt

    rb, rs, h = BIG_RADIUS_UM, SMALL_RADIUS_UM, 0.5 * width
    tb, ts = math.asin(h / rb), math.asin(h / rs)
    xb, xs_center = rb * math.cos(tb), rb + length + rs
    xs = xs_center - rs * math.cos(ts)
    nb = max(64, math.ceil(2 * math.pi * rb / SEGMENT_UM))
    ns = max(48, math.ceil(2 * math.pi * rs / SEGMENT_UM))
    step = min(width / 4, SEGMENT_UM)
    throat_area = min(BODY_AREA_UM2, step**2 * math.sqrt(3) / 4)
    nt = max(8, math.ceil((xs - xb) / step))
    nh = max(2, math.ceil(width / step))
    # Mouth chords: identical, exactly computed vertices shared by the disk and throat polygons.
    # Mismatched or nearly coincident vertices (1e-16 apart) make Triangle refine without end.
    mouth_b = [[xb, h - width * k / nh] for k in range(nh + 1)]  # top -> bottom at x = xb
    mouth_s = [[xs, -h + width * k / nh] for k in range(nh + 1)]  # bottom -> top at x = xs
    big_arc = base.arc(0, 0, rb, tb, 2 * math.pi - tb, nb)[1:-1]  # counter-clockwise from upper to lower mouth
    big = mt.createPolygon(mouth_b[:1] + big_arc + mouth_b[::-1][:-1], isClosed=True, marker=1, area=BODY_AREA_UM2)
    pts = [[xb + (xs - xb) * k / nt, -h] for k in range(nt + 1)] + mouth_s[1:-1]
    pts += [[xs - (xs - xb) * k / nt, h] for k in range(nt + 1)] + mouth_b[1:-1]
    throat = mt.createPolygon(pts, isClosed=True, marker=2, area=throat_area)
    small_arc = base.arc(xs_center, 0, rs, -math.pi + ts, math.pi - ts, ns)[1:-1]  # lower mouth around the right to upper mouth
    small = mt.createPolygon(mouth_s[:1] + small_arc + mouth_s[::-1][:-1], isClosed=True, marker=3, area=BODY_AREA_UM2)
    return mt.mergePLC([big, throat, small])


def period_plc():
    """Full chain period [S/2 - throat - B - throat - S/2]: contains a complete B (question 1 check)."""

    import pygimli.meshtools as mt

    rb, rs = BIG_RADIUS_UM, SMALL_RADIUS_UM
    w, length = base.GEOMETRY_UM["throat_width"], base.GEOMETRY_UM["throat_length"]
    h = 0.5 * w
    tb, ts = math.asin(h / rb), math.asin(h / rs)
    nb = max(64, math.ceil(2 * math.pi * rb / SEGMENT_UM))
    ns = max(24, math.ceil(math.pi * rs / 2 / SEGMENT_UM))
    step = w / 4
    area = step**2 * math.sqrt(3) / 4
    xb = rs + length + rb  # B centre; left cut through S centre at x = 0
    big = mt.createPolygon(
        base.arc(xb, 0, rb, tb, math.pi - tb, nb // 2) + base.arc(xb, 0, rb, math.pi + tb, 2 * math.pi - tb, nb // 2),
        isClosed=True, marker=1, area=BODY_AREA_UM2,
    )
    parts = [big]
    for x0, x1 in ((rs * math.cos(ts), xb - rb * math.cos(tb)), (xb + rb * math.cos(tb), 2 * xb - rs * math.cos(ts))):
        nt = max(8, math.ceil((x1 - x0) / step))
        pts = [[x0 + (x1 - x0) * k / nt, -h] for k in range(nt + 1)] + [[x1 - (x1 - x0) * k / nt, h] for k in range(nt + 1)]
        parts.append(mt.createPolygon(pts, isClosed=True, marker=2, area=area))
    for cx, a0, a1 in ((0.0, -math.pi / 2, -ts), (2 * xb, math.pi + ts, 1.5 * math.pi)):
        if cx == 0.0:
            pts = [[0.0, -rs]] + base.arc(0, 0, rs, -math.pi / 2, -ts, ns)[1:] + base.arc(0, 0, rs, ts, math.pi / 2, ns)
        else:
            pts = base.arc(cx, 0, rs, math.pi + ts, 1.5 * math.pi, ns) + [[cx, rs]] + base.arc(cx, 0, rs, math.pi / 2, math.pi - ts, ns)[1:]
        parts.append(mt.createPolygon(pts, isClosed=True, marker=3, area=BODY_AREA_UM2))
    return mt.mergePLC(parts), 2 * xb


def simulate(mesh):
    n_steps = int(round(base.TIME_MAX_MS / base.TIME_STEP_MS))
    total, regional = base.BlochTorreyStepper(mesh, base.TIME_STEP_MS).run(n_steps)
    return np.arange(n_steps + 1) * base.TIME_STEP_MS, total, regional


def run_case(args):
    width, length = args
    import matplotlib

    matplotlib.use("Agg", force=True)
    start = time.perf_counter()
    mesh = base.build_mesh(closed_plc(width, length))
    times, total, regional = simulate(mesh)
    inv = base.invert(times, total, f"w{width}_L{length}", np.random.default_rng(base.NOISE_SEED))
    lc = inv["lcurve"]
    areas = {m: float(np.asarray(mesh.cellSizes())[np.array([c.marker() for c in mesh.cells()]) == m].sum()) for m in (1, 2, 3)}
    return {
        "width_um": width,
        "length_um": length,
        "nodes": int(mesh.nodeCount()),
        "area_big_um2": areas[1],
        "area_throat_um2": areas[2],
        "area_small_um2": areas[3],
        "t2_big_region_fit_ms": base.fit_t2_ms(times, regional[1], 0.9, 0.1),
        "t2_small_region_fit_ms": base.fit_t2_ms(times, regional[3], 0.9, 0.1),
        "alpha": float(lc.best_regularization),
        "t2_bins_ms": lc.t2_bins_ms.tolist(),
        "spectrum": lc.spectrum.tolist(),
        "decay_every_1ms": (total[::4] / total[0]).tolist(),
        # Full echo train (dt = TIME_STEP_MS, noise-free, normalized) so other inversions
        # (e.g. MATLAB BRD) can rebuild exactly the same noisy input with NOISE_SEED.
        "decay_full_normalized": (total / total[0]).tolist(),
        "seconds": time.perf_counter() - start,
    }


def peak_metrics(t2, spectrum):
    """Peaks (>2% amplitude fraction), and the valley depth between the two largest peaks."""

    t2, a = np.asarray(t2), np.asarray(spectrum)
    peaks = [p for p in base.spectrum_peaks(t2, a) if p["fraction"] > 0.02 and p["position"] > 5.0]
    out = {"n_peaks": len(peaks), "peaks": peaks}
    if len(peaks) >= 2:
        top = sorted(sorted(peaks, key=lambda p: -p["fraction"])[:2], key=lambda p: p["position"])
        i0, i1 = (int(np.argmin(np.abs(t2 - p["position"]))) for p in top)
        valley = a[i0 : i1 + 1].min()
        out.update(
            small_peak_t2_ms=top[0]["position"],
            small_peak_fraction=top[0]["fraction"],
            big_peak_t2_ms=top[1]["position"],
            valley_to_lower_peak=float(valley / min(a[i0], a[i1])),
        )
    else:
        main = max(peaks, key=lambda p: p["fraction"]) if peaks else {"position": np.nan}
        out.update(small_peak_t2_ms=np.nan, small_peak_fraction=0.0, big_peak_t2_ms=main["position"], valley_to_lower_peak=1.0)
    return out


def check_symmetry(out: Path):
    """Question 1: half cell vs the complete chain period vs the closed single-throat geometry."""

    import pygimli.meshtools as mt  # noqa: F401

    plc, xs_center = base.half_cell_plc({"body_area_um2": BODY_AREA_UM2, "throat_cells_across": 4, "segment_um": SEGMENT_UM})
    half = base.build_mesh(plc, symmetry_x=(0.0, xs_center))
    plc, x_right = period_plc()
    period = base.build_mesh(plc, symmetry_x=(0.0, x_right))
    closed = base.build_mesh(closed_plc(base.GEOMETRY_UM["throat_width"], base.GEOMETRY_UM["throat_length"]))
    rows, decays = [], {}
    for name, mesh in (("chain_half_cell", half), ("chain_full_period", period), ("closed_single_throat", closed)):
        times, total, regional = simulate(mesh)
        decays[name] = total / total[0]
        rows.append({
            "geometry": name,
            "nodes": int(mesh.nodeCount()),
            "pore_area_um2": float(np.sum(mesh.cellSizes())),
            "t2_big_region_fit_ms": base.fit_t2_ms(times, regional[1], 0.9, 0.1),
            "t2_small_region_fit_ms": base.fit_t2_ms(times, regional[3], 0.9, 0.1),
        })
    frame = pd.DataFrame(rows)
    frame["max_abs_decay_diff_vs_half_cell"] = [float(np.max(np.abs(decays[g] - decays["chain_half_cell"]))) for g in frame["geometry"]]
    frame.to_csv(out / "question1_symmetry_check.csv", index=False)
    print(frame.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "simulation_outputs" / "ideal_two_pore_throat_sweep_nmr_20261012")
    parser.add_argument("--check-symmetry", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=600.0, help="kill and skip a case that runs longer")
    parser.add_argument("--single", type=float, nargs=2, metavar=("WIDTH_UM", "LENGTH_UM"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    case_dir = out / "cases"
    case_dir.mkdir(exist_ok=True)

    def case_path(case):
        return case_dir / f"w{case[0]:g}_L{case[1]:g}.json"

    if args.check_symmetry:
        check_symmetry(out)
        return
    if args.single:
        # Child process: run one case and save it immediately.
        case = tuple(args.single)
        result = run_case(case)
        tmp = case_path(case).with_suffix(".tmp")
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(case_path(case))
        return

    # Parent: one child process per case, saved as soon as it finishes; cases over the time limit are killed and skipped.
    cases = [(w, L) for w in WIDTHS_UM for L in LENGTHS_UM]
    pending = [case for case in cases if not case_path(case).exists()]
    log = out / "sweep_run.log"
    print(f"{len(cases) - len(pending)} cases already saved, {len(pending)} to run", flush=True)

    def launch(case):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(out), "--single", str(case[0]), str(case[1])]
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout_s)
            ok = proc.returncode == 0 and case_path(case).exists()
            status = "saved" if ok else f"failed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}"
        except subprocess.TimeoutExpired:
            status = f"skipped: exceeded {args.timeout_s:.0f} s"
        line = f"w={case[0]:g} L={case[1]:g}: {status} ({time.perf_counter() - start:.0f} s)"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return case, status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = dict(pool.map(launch, pending))
    results = [json.loads(case_path(case).read_text(encoding="utf-8")) for case in cases if case_path(case).exists()]
    (out / "sweep_raw.json").write_text(json.dumps(results), encoding="utf-8")
    missing = [f"w={c[0]:g} L={c[1]:g}: {statuses.get(c, 'not run')}" for c in cases if not case_path(c).exists()]
    print(json.dumps({"status": "complete" if not missing else "incomplete", "saved": len(results), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
