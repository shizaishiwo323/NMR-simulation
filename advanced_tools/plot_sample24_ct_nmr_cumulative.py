"""Plot sample24 CT pore-volume and first-test NNLS cumulative curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(r"E:\python\jupyter\codex\震电模拟\Wang_SE_experiment")
CT_DIR = EXPERIMENT / "CT/Post-24-1075μm"
NMR_DIR = EXPERIMENT / "NMR/data analysis/sample24_first_test_T2_comparison_20261011"


def cumulative(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    if not valid.any():
        raise ValueError("No positive finite weighted observations")
    order = np.argsort(values[valid], kind="stable")
    x, mass = values[valid][order], weights[valid][order]
    cdf = np.cumsum(mass) / mass.sum()
    cdf[-1] = 1.0
    return x, cdf


def log_quantile(x: np.ndarray, cdf: np.ndarray, q: float) -> float:
    if not cdf[0] <= q <= 1:
        raise ValueError("Alignment outside the measured cumulative curve")
    if q == 1:
        return float(x[-1])
    return float(10 ** np.interp(q, cdf, np.log10(x)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization", choices=("independent", "nmr_total"), required=True)
    parser.add_argument("--alignment-vc", nargs="+", type=float, default=[1, 0.5])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pore_path = CT_DIR / "pore_network/network_parsed/pores.csv"
    spectrum_path = NMR_DIR / "nnls_spectrum.csv"
    porosity_path = NMR_DIR / "sample24_first_porosity_comparison.csv"
    ct_manifest_path = CT_DIR / "pore_network/post24_4_24_fullres_run_summary.json"
    nnls_summary_path = NMR_DIR / "nnls_summary.json"
    pores, spectrum = pd.read_csv(pore_path), pd.read_csv(spectrum_path)
    radius, ct_cdf = cumulative(pores.pore_radius_m.to_numpy(float) * 1e6,
                                pores.pore_volume_m3.to_numpy(float))
    t2_ms, nmr_cdf = cumulative(spectrum.t2_ms.to_numpy(float), spectrum.amplitude_au.to_numpy(float))
    ct_manifest = json.loads(ct_manifest_path.read_text(encoding="utf-8"))
    ct_porosity = float(ct_manifest["prepare"]["porosity"])
    porosity = pd.read_csv(porosity_path)
    nmr_porosity = float(porosity.loc[porosity.method == "NNLS_Lcurve", "porosity_pct"].iloc[0]) / 100
    nnls = json.loads(nnls_summary_path.read_text(encoding="utf-8"))
    baseline = 1 - ct_porosity / nmr_porosity
    if args.normalization == "nmr_total":
        if not 0 <= baseline < 1:
            raise ValueError("CT porosity must be positive and no greater than NMR total porosity")
        ct_vc = baseline + (1 - baseline) * ct_cdf
    else:
        ct_vc = ct_cdf

    alignments = []
    for vc in args.alignment_vc:
        ct_anchor = log_quantile(radius, ct_vc, vc)
        t2_anchor = log_quantile(t2_ms, nmr_cdf, vc)
        rho2 = ct_anchor / (3 * t2_anchor * 1e-3)
        mapped_radius = 0.003 * rho2 * t2_ms
        assert np.isclose(log_quantile(mapped_radius, nmr_cdf, vc), ct_anchor)
        assert np.all(np.diff(ct_vc) >= 0) and np.all(np.diff(nmr_cdf) >= 0)
        assert np.isclose(ct_vc[-1], 1) and np.isclose(nmr_cdf[-1], 1)

        token = f"{vc:g}".replace(".", "p")
        source = args.output_dir / f"sample24_ct_nmr_cumulative_vc_{token}_source_data.csv"
        pd.concat([
            pd.DataFrame({"method": "CT pore body", "radius_um": radius,
                          "cumulative_vc": ct_vc, "raw_resolved_cdf": ct_cdf}),
            pd.DataFrame({"method": "NMR first-test NNLS", "radius_um": mapped_radius,
                          "cumulative_vc": nmr_cdf, "t2_ms": t2_ms}),
        ], ignore_index=True).to_csv(source, index=False)

        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                             "font.size": 9, "axes.linewidth": 0.8})
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot(radius, ct_vc, color="#285f64", linewidth=1.9, label="CT pore body")
        ax.plot(mapped_radius, nmr_cdf, color="#4c65ae", linewidth=1.9, linestyle="--",
                label=rf"NMR NNLS ($V_c={vc:g}$; $\rho_2^*={rho2:.2f}$ $\mu$m/s)")
        ax.set_xscale("log")
        ax.set_xlim(min(radius[0], mapped_radius[0]) / 1.3,
                    max(radius[-1], mapped_radius[-1]) * 1.25)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"Equivalent pore radius, $r$ ($\mu$m)")
        ax.set_ylabel(r"Cumulative pore volume, $V_c$")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(which="both", direction="out", width=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout()
        figure = args.output_dir / f"sample24_ct_nmr_cumulative_vc_{token}.png"
        fig.savefig(figure, dpi=600, facecolor="white", bbox_inches="tight")
        assert len(ax.texts) == 0 and ax.get_title() == ""
        plt.close(fig)
        alignments.append({"vc": vc, "rho2_effective_um_per_s": rho2,
                           "ct_anchor_radius_um": ct_anchor, "nmr_anchor_t2_ms": t2_anchor,
                           "nmr_mapped_radius_range_um": [float(mapped_radius[0]), float(mapped_radius[-1])],
                           "figure": str(figure.resolve()), "source_data": str(source.resolve())})

    inputs = [pore_path, spectrum_path, porosity_path, ct_manifest_path, nnls_summary_path]
    manifest = {
        "sample": "24", "nmr_test": "first", "nmr_reinverted": False,
        "nnls_alpha": nnls["best_alpha"], "normalization": args.normalization,
        "ct_porosity": ct_porosity, "nmr_total_porosity": nmr_porosity,
        "unresolved_fraction_if_total_porosity_corrected": baseline,
        "ct_cumulative_formula": "1-f+f*F_CT" if args.normalization == "nmr_total" else "F_CT=sum(V_i for r_i<=r)/sum(V_i)",
        "nmr_cumulative_formula": "F_NMR=sum(A_i for T2_i<=T2)/sum(A_i)",
        "mapping": "r[um]=0.003*rho2*[um/s]*T2[ms]; spherical fast-diffusion equivalent radius; bulk term omitted",
        "calibration": "rho2*=CT_radius_at_Vc/(3*NMR_T2_at_Vc[s]); log-coordinate interpolation",
        "boundary": "rho2* is a conditional alignment coefficient; independent normalization compares distribution shape and does not reconcile absolute CT/NMR porosity",
        "ct_pores_used": len(radius), "ct_pores_total": len(pores), "ct_network": "full-resolution, not visualization LOD",
        "ct_radius_range_um": [float(radius[0]), float(radius[-1])],
        "nmr_positive_t2_range_ms": [float(t2_ms[0]), float(t2_ms[-1])],
        "trimming": "No visual plateau trimming; exclude only nonpositive/nonfinite radii, volumes or amplitudes",
        "alignments": alignments,
        "figure_contract": {"conclusion": "Compare CT and experimental NMR cumulative shapes under two selected volume alignments",
                            "archetype": "quantitative grid", "backend": "Python/matplotlib",
                            "export": "PNG, 600 dpi", "statistics": "One sample; descriptive comparison only",
                            "visible_text": "Axis labels, ticks and legend only; no title or annotations"},
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(alignments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    x, cdf = cumulative(np.array([10., 1., 5.]), np.array([1., 2., 1.]))
    assert np.allclose(cdf, [0.5, 0.75, 1.0]) and log_quantile(x, cdf, 0.5) == 1
    main()
