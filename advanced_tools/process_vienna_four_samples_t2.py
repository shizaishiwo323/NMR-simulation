"""Compare T2process L-curve NNLS and MATLAB BRD for four Vienna samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_IDS = ("24", "25", "26", "34")
TESTS = ("first", "second")
DIMENSIONS_AND_MASS_G = {
    "24": (18.90, 18.40, 26.00, 14.88, 17.75),
    "25": (18.00, 15.22, 22.76, 10.48, 12.53),
    "26": (22.30, 19.80, 19.90, 14.49, 17.06),
    "34": (22.20, 21.40, 18.40, 13.52, 16.04),
}
REFERENCE_POROSITY_PERCENT = {"24": 38.2, "25": 37.4, "26": 34.2, "34": 35.6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--t2process-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_decay(data_root: Path, sample_id: str, test: str) -> tuple[np.ndarray, np.ndarray, Path]:
    name = f"{sample_id}.1.pea" if test == "first" else f"M14{sample_id}.1.pea"
    path = data_root / f"{test} test" / name
    raw = np.loadtxt(path, skiprows=4)
    valid = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & (raw[:, 0] > 0)
    time_ms, signal = raw[valid, 0], raw[valid, 1]
    order = np.argsort(time_ms)
    return time_ms[order], signal[order], path


def peak_area_table(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="peak area", header=1)
    frame["label"] = frame["Sampling remarks"].astype(str).str.replace("\\t", "", regex=False).str.strip()
    return frame


def peak_area(frame: pd.DataFrame, label: str, occurrence: int = 0) -> float:
    values = frame.loc[frame["label"].eq(label), "Sum of peak areas"].astype(float).to_numpy()
    if values.size <= occurrence:
        raise ValueError(f"Missing peak-area row: {label}")
    return float(values[occurrence])


def calculate_porosity(data_root: Path) -> tuple[pd.DataFrame, dict]:
    first = peak_area_table(data_root / "first test" / "PeakArea-S30-1-20260705151050.xlsx")
    first_34 = peak_area_table(data_root / "first test" / "PeakArea-34-1-20260705151209.xlsx")
    second = peak_area_table(data_root / "second test" / "PeakArea-S302-1-20260706141927.xlsx")
    areas = {
        "first": {"standard": peak_area(first, "Standard_30%-1")},
        "second": {"standard": peak_area(second, "Standard30-1")},
    }
    for sample_id in SAMPLE_IDS:
        areas["first"][sample_id] = peak_area(first_34, "34-1") if sample_id == "34" else peak_area(first, f"{sample_id}-1")
        areas["second"][sample_id] = peak_area(second, f"M14{sample_id}-1")

    rows = []
    calibration = {}
    water_g = np.array([DIMENSIONS_AND_MASS_G[s][4] - DIMENSIONS_AND_MASS_G[s][3] for s in SAMPLE_IDS])
    for test in TESTS:
        sample_areas = np.array([areas[test][s] for s in SAMPLE_IDS])
        area_per_g = float(np.dot(water_g, sample_areas) / np.dot(water_g, water_g))
        standard_water_equivalent_ml = float(areas[test]["standard"] / area_per_g)
        calibration[test] = {
            "area_per_g_water_zero_intercept": area_per_g,
            "standard_peak_area": areas[test]["standard"],
            "standard_water_equivalent_ml": standard_water_equivalent_ml,
            "standard_porosity_fraction_from_name": 0.30,
            "inferred_standard_bulk_volume_ml": standard_water_equivalent_ml / 0.30,
        }
        for sample_id, area in zip(SAMPLE_IDS, sample_areas):
            d1, d2, length, dry_g, wet_g = DIMENSIONS_AND_MASS_G[sample_id]
            bulk_volume_ml = np.pi * d1 * d2 * length / 4000.0
            nmr_water_ml = area / area_per_g
            rows.append(
                {
                    "sample_id": f"M14/{sample_id}",
                    "test": test,
                    "bulk_volume_ml_elliptical_cylinder": bulk_volume_ml,
                    "dry_mass_g": dry_g,
                    "saturated_mass_g": wet_g,
                    "gravimetric_water_ml_rho1": wet_g - dry_g,
                    "instrument_total_peak_area": area,
                    "nmr_water_ml_mass_calibrated": nmr_water_ml,
                    "nmr_porosity_percent": 100.0 * nmr_water_ml / bulk_volume_ml,
                    "gravimetric_porosity_percent": 100.0 * (wet_g - dry_g) / bulk_volume_ml,
                    "reference_ppt_porosity_percent": REFERENCE_POROSITY_PERCENT[sample_id],
                }
            )
    detailed = pd.DataFrame(rows)
    summary = detailed.pivot(index="sample_id", columns="test", values="nmr_porosity_percent").reset_index()
    summary.columns.name = None
    summary = summary.rename(columns={"first": "nmr_porosity_first_percent", "second": "nmr_porosity_second_percent"})
    summary["nmr_porosity_mean_percent"] = summary[["nmr_porosity_first_percent", "nmr_porosity_second_percent"]].mean(axis=1)
    summary["repeat_absolute_difference_percent_points"] = (
        summary["nmr_porosity_first_percent"] - summary["nmr_porosity_second_percent"]
    ).abs()
    grav = detailed.groupby("sample_id", as_index=False)[
        ["bulk_volume_ml_elliptical_cylinder", "gravimetric_porosity_percent", "reference_ppt_porosity_percent"]
    ].first()
    return summary.merge(grav, on="sample_id"), {"tests": calibration, "detailed_rows": rows}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.t2process_root.resolve()))
    from nmr_t2 import LCurveConfig
    from nmr_t2.lcurve import invert_single_signal_lcurve

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    brd_summary = pd.read_csv(output / "brd_summary.csv")
    lcurve_config = LCurveConfig(alpha_min=1e-4, alpha_max=1e5, alpha_count=80)
    spectrum_rows: list[dict] = []
    result_rows: list[dict] = []
    metric_rows: list[dict] = []

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.2), sharex=True, sharey=True)
    colors = {"L-curve NNLS": "#222222", "BRD 0.1": "#3B82B4", "BRD 0.5": "#D88C3A", "BRD 1": "#8C6BB1"}

    for row_index, sample_id in enumerate(SAMPLE_IDS):
        for col_index, test in enumerate(TESTS):
            axis = axes[row_index, col_index]
            time_ms, signal, source_path = read_decay(args.data_root, sample_id, test)
            normalized_signal = signal / signal[0]
            lcurve = invert_single_signal_lcurve(
                time_ms,
                normalized_signal,
                signal_name=f"sample{sample_id}_{test}",
                config=lcurve_config,
            )
            lcurve_spectrum = np.asarray(lcurve.spectrum, dtype=float)
            lcurve_peak_norm = lcurve_spectrum / lcurve_spectrum.max()
            relative_residual = float(np.linalg.norm(lcurve.residual) / np.linalg.norm(lcurve.fit_amplitude + lcurve.residual))
            peak_index = int(np.argmax(lcurve_spectrum))
            axis.plot(lcurve.t2_bins_ms, lcurve_peak_norm, color=colors["L-curve NNLS"], lw=1.6, label="L-curve NNLS")
            result_rows.append(
                {
                    "sample_id": f"M14/{sample_id}",
                    "test": test,
                    "method": "L-curve NNLS",
                    "regularization": lcurve.best_regularization,
                    "peak_t2_ms": lcurve.t2_bins_ms[peak_index],
                    "relative_residual": relative_residual,
                    "first_echo_ms": time_ms[0],
                    "first_echo_amplitude": signal[0],
                    "source_decay": str(source_path.resolve()),
                }
            )
            for t2, raw_amp, normalized_amp in zip(lcurve.t2_bins_ms, lcurve_spectrum, lcurve_peak_norm):
                spectrum_rows.append(
                    {"sample_id": f"M14/{sample_id}", "test": test, "method": "L-curve NNLS", "regularization": lcurve.best_regularization,
                     "t2_ms": t2, "raw_spectrum_amplitude": raw_amp, "peak_normalized_amplitude": normalized_amp}
                )
            for index, alpha in enumerate(lcurve.alpha_values):
                metric_rows.append(
                    {"sample_id": f"M14/{sample_id}", "test": test, "alpha": alpha, "residual_norm": lcurve.residual_norms[index],
                     "roughness_norm": lcurve.roughness_norms[index], "reciprocal_slope": lcurve.slope_reciprocal_values[index],
                     "is_selected": index == lcurve.best_index}
                )

            brd = pd.read_csv(output / f"sample{sample_id}_{test}_brd_spectra.csv")
            for alpha, column in ((0.1, "alpha_0p1"), (0.5, "alpha_0p5"), (1.0, "alpha_1")):
                raw_spectrum = brd[column].to_numpy(float)
                peak_norm = raw_spectrum / raw_spectrum.max()
                label = f"BRD {alpha:g}"
                axis.plot(brd["t2_ms"], peak_norm, color=colors[label], lw=1.25, alpha=0.9, label=label)
                row = brd_summary[
                    (brd_summary["sample_id"].astype(str) == sample_id)
                    & (brd_summary["test"] == test)
                    & np.isclose(brd_summary["alpha"], alpha)
                ].iloc[0]
                result_rows.append(
                    {"sample_id": f"M14/{sample_id}", "test": test, "method": "MATLAB BRD", "regularization": alpha,
                     "peak_t2_ms": row["peak_t2_ms"], "relative_residual": row["relative_residual"],
                     "first_echo_ms": time_ms[0], "first_echo_amplitude": signal[0], "source_decay": str(source_path.resolve())}
                )
                for t2, raw_amp, normalized_amp in zip(brd["t2_ms"], raw_spectrum, peak_norm):
                    spectrum_rows.append(
                        {"sample_id": f"M14/{sample_id}", "test": test, "method": "MATLAB BRD", "regularization": alpha,
                         "t2_ms": t2, "raw_spectrum_amplitude": raw_amp, "peak_normalized_amplitude": normalized_amp}
                    )

            axis.set_xscale("log")
            axis.set_xlim(1e-2, 1e5)
            axis.set_ylim(0, 1.05)
            axis.grid(which="both", color="#D8D8D8", lw=0.45, alpha=0.6)
            axis.set_title(f"M14/{sample_id}, {test} test", fontsize=8)
            if col_index == 0:
                axis.set_ylabel("Peak-normalized amplitude")
            if row_index == len(SAMPLE_IDS) - 1:
                axis.set_xlabel(r"$T_2$ (ms)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995), fontsize=7)
    fig.suptitle("Vienna carbonate samples: T2 inversion sensitivity", y=1.018, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.0, w_pad=1.0)
    figure_path = output / "vienna_four_samples_t2_lcurve_brd.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(spectrum_rows).to_csv(output / "t2_spectra_long.csv", index=False)
    result_frame = pd.DataFrame(result_rows)
    metric_frame = pd.DataFrame(metric_rows)
    result_frame.to_csv(output / "t2_inversion_summary.csv", index=False)
    metric_frame.to_csv(output / "lcurve_metrics.csv", index=False)
    selected_nnls = metric_frame.loc[metric_frame["is_selected"]].merge(
        result_frame.loc[result_frame["method"] == "L-curve NNLS"],
        left_on=["sample_id", "test", "alpha"],
        right_on=["sample_id", "test", "regularization"],
        validate="one_to_one",
    )
    selected_nnls[
        [
            "sample_id", "test", "alpha", "residual_norm", "roughness_norm",
            "reciprocal_slope", "peak_t2_ms", "relative_residual", "first_echo_ms",
            "first_echo_amplitude", "source_decay",
        ]
    ].rename(columns={"alpha": "selected_regularization"}).to_csv(
        output / "nnls_lcurve_selected_parameters.csv", index=False
    )
    porosity, calibration = calculate_porosity(args.data_root)
    porosity.to_csv(output / "porosity_summary.csv", index=False)
    manifest = {
        "t2process_commit": "47461a72e0852e5d2a5a08db19a1f8404145e236",
        "samples": [f"M14/{sample_id}" for sample_id in SAMPLE_IDS],
        "tests": list(TESTS),
        "lcurve": {**lcurve_config.__dict__, "signal_normalization": "divide by first echo amplitude"},
        "brd": {"solver": "T2process brd/matlab/brd1.p", "alpha_values": [0.1, 0.5, 1.0], "num_t2_bins": 200,
                "t2_range_ms": [0.01, 100000.0], "echo_sampling": "T2process AmplitudeSample, 200 points"},
        "porosity": {
            "bulk_volume": "elliptical cylinder pi*d1*d2*length/4 from the laboratory record photograph",
            "water_density_g_per_ml": 1.0,
            "calibration": "zero-intercept fit of instrument total peak area versus gravimetric pore-water mass across the four saturated samples",
            "independence_caveat": "Absolute NMR porosity is mass-calibrated and is not independent of the wet-minus-dry mass measurements.",
            **calibration,
        },
        "figure_contract": {
            "conclusion": "Peak location and secondary structure are compared across L-curve NNLS and three fixed BRD smoothing factors for both repeat tests.",
            "archetype": "quantitative grid",
            "technical_repeats": 2,
            "statistics": "No inferential statistics; the two tests are shown separately.",
            "export": "PNG, 600 dpi",
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    assert len(result_rows) == 32 and len(spectrum_rows) == 6400 and len(selected_nnls) == 8
    print(porosity.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(figure_path)


if __name__ == "__main__":
    main()
