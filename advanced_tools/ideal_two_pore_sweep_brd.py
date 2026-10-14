"""Re-invert the throat-sweep NMR decays with the T2process MATLAB BRD solver at a fixed alpha.

Input decays are rebuilt exactly as in the L-curve version: full noise-free echo train
(dt = 0.25 ms) + Gaussian noise (sigma = NOISE_RELATIVE, seed NOISE_SEED). BRD steps follow
sample16_brd_alpha_sweep.m (first-echo normalization, AmplitudeSample 200 points,
T2 grid logspace(-2, 5, 200) ms, brd1).

    python advanced_tools/ideal_two_pore_sweep_brd.py --alpha 0.1
    python advanced_tools/ideal_two_pore_nature_plots.py sweep --sweep-dir <dir> --raw sweep_raw_brd_alpha0p1.json --method "BRD (α = 0.1)" --tag _brd_alpha0p1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import advanced_tools.ideal_two_pore_channel_nmr_sip as base  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, default=ROOT / "simulation_outputs" / "ideal_two_pore_throat_sweep_nmr_brd_20261013")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--brd-dir", type=Path, default=ROOT / "tmp" / "T2process_20261013" / "brd" / "matlab")
    parser.add_argument("--matlab", default=r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe")
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    tag = f"brd_alpha{args.alpha:g}".replace(".", "p")
    input_dir, output_dir = sweep_dir / f"{tag}_input", sweep_dir / f"{tag}_output"
    input_dir.mkdir(exist_ok=True)

    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((sweep_dir / "cases").glob("*.json"))]
    stems = {}
    for case in cases:
        clean = np.asarray(case["decay_full_normalized"])
        times = np.arange(clean.size) * base.TIME_STEP_MS
        noisy = clean + np.random.default_rng(base.NOISE_SEED).normal(0.0, base.NOISE_RELATIVE, clean.size)
        stem = f"w{case['width_um']:g}_L{case['length_um']:g}"
        stems[stem] = case
        np.savetxt(input_dir / f"{stem}.csv", np.column_stack([times[1:], noisy[1:]]), delimiter=",")

    call = f"addpath('{(ROOT / 'advanced_tools').as_posix()}'); ideal_two_pore_brd_batch('{input_dir.as_posix()}', '{output_dir.as_posix()}', '{args.brd_dir.resolve().as_posix()}', {args.alpha!r});"
    log = subprocess.run([args.matlab, "-batch", call], capture_output=True, text=True, encoding="utf-8", errors="replace")
    (sweep_dir / f"{tag}_matlab.log").write_text(log.stdout + log.stderr, encoding="utf-8")
    if log.returncode != 0:
        raise RuntimeError(f"MATLAB BRD failed:\n{log.stderr[-2000:]}")

    raw = []
    for stem, case in stems.items():
        table = np.loadtxt(output_dir / f"{stem}_brd.csv", delimiter=",")
        entry = {k: v for k, v in case.items() if k not in ("decay_full_normalized", "t2_bins_ms", "spectrum", "alpha")}
        entry.update(t2_bins_ms=table[:, 0].tolist(), spectrum=table[:, 1].tolist(), alpha=args.alpha)
        raw.append(entry)
    (sweep_dir / f"sweep_raw_{tag}.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest = {
        "method": "T2process MATLAB brd1.p (P-code), fixed alpha",
        "alpha": args.alpha,
        "brd_dir": str(args.brd_dir.resolve()),
        "brd1_p_sha256": hashlib.sha256((args.brd_dir / "brd1.p").read_bytes()).hexdigest(),
        "t2process_repo": "https://github.com/shizaishiwo323/T2process (commit 47461a7)",
        "steps": "first-echo normalization; AmplitudeSample(t, y, 200, 0); T2 grid logspace(-2, 5, 200) ms; kernel exp(-t/T2); brd1(kernel, y, alpha)",
        "input_noise": {"relative_sigma": base.NOISE_RELATIVE, "seed": base.NOISE_SEED, "echo_spacing_ms": base.TIME_STEP_MS},
        "cases": len(raw),
        "matlab_log": str(sweep_dir / f"{tag}_matlab.log"),
    }
    (sweep_dir / f"{tag}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": len(raw), "raw": f"sweep_raw_{tag}.json"}))


if __name__ == "__main__":
    main()
