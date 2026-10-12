"""Overlay Vienna-sample NNLS T2 spectra with shared-volume PNM radius histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NMR_DIR = Path(r"E:\python\jupyter\codex\NMR模拟\simulation_outputs\vienna_m14_four_samples_t2_20260921")
CT_BASE = Path(r"E:\python\jupyter\codex\震电模拟\Wang_SE_experiment\CT")
N_BINS = 30
T2_LIMITS_MS = (0.05, 5000.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nmr_spectra(
    sample_id: str,
    tests: tuple[str, ...],
    spectrum_override: Path | None,
    summary_override: Path | None,
    spectrum_table: Path | None,
    parameters_table: Path | None,
) -> tuple[pd.DataFrame, dict]:
    if (spectrum_table is None) != (parameters_table is None):
        raise ValueError("--nmr-spectrum-table and --nmr-parameters-table must be supplied together")
    if spectrum_table is not None and (spectrum_override is not None or summary_override is not None):
        raise ValueError("use either NMR tables or the single-spectrum files, not both")
    if (spectrum_override is None) != (summary_override is None):
        raise ValueError("--nmr-spectrum and --nmr-summary must be supplied together")
    if spectrum_table is not None:
        sample_number = int(sample_id.split("/")[-1])
        all_spectra = pd.read_csv(spectrum_table)
        parameters = pd.read_csv(parameters_table)
        selected = all_spectra[
            (all_spectra["sample"] == sample_number)
            & (all_spectra["test"].isin(tests))
            & (all_spectra["method"] == "NNLS (L-curve)")
        ].copy()
        selected_parameters = parameters[
            (parameters["sample"] == sample_number) & (parameters["test"].isin(tests))
        ][["test", "alpha"]].copy()
        if selected_parameters["test"].duplicated().any():
            raise ValueError("NNLS parameter table contains duplicate sample/test rows")
        spectra = selected.merge(selected_parameters, on="test", how="left", validate="many_to_one")
        if spectra["alpha"].isna().any():
            raise ValueError("NNLS alpha is missing for one or more requested spectra")
        spectra = spectra.rename(
            columns={"alpha": "regularization", "amplitude_au": "raw_spectrum_amplitude"}
        )
        spectra["sample_id"] = sample_id
        provenance = {
            "mode": "authoritative_four_sample_tables",
            "spectrum_table_path": str(spectrum_table),
            "spectrum_table_sha256": sha256(spectrum_table),
            "parameters_table_path": str(parameters_table),
            "parameters_table_sha256": sha256(parameters_table),
        }
    elif spectrum_override is None:
        source = NMR_DIR / "t2_spectra_long.csv"
        spectra = pd.read_csv(source)
        spectra = spectra[
            (spectra["sample_id"] == sample_id) & (spectra["method"] == "L-curve NNLS")
        ].copy()
        spectra = spectra[spectra["test"].isin(tests)].copy()
        provenance = {
            "mode": "legacy_batch_table",
            "spectrum_path": str(source),
            "spectrum_sha256": sha256(source),
        }
    else:
        if len(tests) != 1:
            raise ValueError("an explicit NNLS spectrum can represent exactly one requested test")
        raw = pd.read_csv(spectrum_override)
        required = {"t2_ms", "amplitude_au"}
        if not required.issubset(raw.columns):
            raise ValueError(f"explicit NNLS spectrum must contain columns {sorted(required)}")
        summary = json.loads(summary_override.read_text(encoding="utf-8"))
        spectra = pd.DataFrame(
            {
                "sample_id": sample_id,
                "test": tests[0],
                "method": "L-curve NNLS",
                "regularization": float(summary["best_alpha"]),
                "t2_ms": raw["t2_ms"].to_numpy(float),
                "raw_spectrum_amplitude": raw["amplitude_au"].to_numpy(float),
            }
        )
        calculated_peak = float(
            spectra.loc[spectra["raw_spectrum_amplitude"].idxmax(), "t2_ms"]
        )
        if not np.isclose(calculated_peak, float(summary["peak_t2_ms"])):
            raise ValueError("NNLS spectrum peak does not match its summary")
        provenance = {
            "mode": "explicit_spectrum_and_summary",
            "spectrum_path": str(spectrum_override),
            "spectrum_sha256": sha256(spectrum_override),
            "summary_path": str(summary_override),
            "summary_sha256": sha256(summary_override),
            "solver": summary.get("solver"),
            "signal_scaling": summary.get("signal_scaling"),
            "inversion_config": summary.get("config"),
        }
    if set(spectra["test"].unique()) != set(tests):
        raise ValueError(f"requested NNLS tests are not all available: {tests}")
    return spectra, provenance


def log_position(value: float, limits: tuple[float, float]) -> float:
    low, high = np.log10(limits)
    return float((np.log10(value) - low) / (high - low))


def nice_log_upper(value: float) -> float:
    decade = 10.0 ** np.floor(np.log10(value))
    for factor in (1.0, 2.0, 5.0, 10.0):
        candidate = factor * decade
        if candidate >= value:
            return float(candidate)
    raise AssertionError("unreachable")


def render_sample(
    sample_number: int,
    *,
    tests: tuple[str, ...] = ("first", "second"),
    network_dir_override: Path | None = None,
    output_dir_override: Path | None = None,
    source_ct: Path | None = None,
    network_extraction_summary: Path | None = None,
    nmr_peak_radius_um: float | None = None,
    nmr_spectrum: Path | None = None,
    nmr_summary: Path | None = None,
    nmr_spectrum_table: Path | None = None,
    nmr_parameters_table: Path | None = None,
) -> dict:
    sample_id = f"M14/{sample_number}"
    network_dir = network_dir_override or CT_BASE / f"Post-{sample_number}-1075μm" / "pore_network" / "network_parsed"
    output_dir = output_dir_override or network_dir.parent / f"sample{sample_number}_nnls_pore_throat_overlay"
    spectra, nmr_provenance = load_nmr_spectra(
        sample_id, tests, nmr_spectrum, nmr_summary, nmr_spectrum_table, nmr_parameters_table
    )
    pores = pd.read_csv(network_dir / "pores.csv")
    throats_all = pd.read_csv(network_dir / "throats.csv")
    internal = (throats_all["pore1_id"] > 0) & (throats_all["pore2_id"] > 0)
    throats = throats_all.loc[internal].copy()

    pore_radius = pores["pore_radius_m"].to_numpy(float) * 1e6
    pore_volume = pores["pore_volume_m3"].to_numpy(float)
    throat_radius = throats["throat_radius_m"].to_numpy(float) * 1e6
    throat_volume = throats["throat_volume_m3"].to_numpy(float)
    pore_valid = np.isfinite(pore_radius) & np.isfinite(pore_volume) & (pore_radius > 0) & (pore_volume > 0)
    throat_valid = np.isfinite(throat_radius) & np.isfinite(throat_volume) & (throat_radius > 0) & (throat_volume > 0)
    pore_radius, pore_volume = pore_radius[pore_valid], pore_volume[pore_valid]
    throat_radius, throat_volume = throat_radius[throat_valid], throat_volume[throat_valid]

    edges = np.logspace(
        np.log10(min(pore_radius.min(), throat_radius.min())),
        np.log10(max(pore_radius.max(), throat_radius.max())),
        N_BINS + 1,
    )
    edges[0] *= 1.0 - 1e-12
    edges[-1] *= 1.0 + 1e-12
    centers = np.sqrt(edges[:-1] * edges[1:])
    pore_bin_volume = np.histogram(pore_radius, bins=edges, weights=pore_volume)[0]
    throat_bin_volume = np.histogram(throat_radius, bins=edges, weights=throat_volume)[0]
    shared_network_volume = float(pore_volume.sum() + throat_volume.sum())
    pore_fraction = pore_bin_volume / shared_network_volume
    throat_fraction = throat_bin_volume / shared_network_volume
    combined_fraction = pore_fraction + throat_fraction
    radius_peak_um = float(centers[np.argmax(combined_fraction)])

    t2_peaks = {}
    for test, group in spectra.groupby("test", sort=False):
        t2_peaks[test] = float(group.loc[group["raw_spectrum_amplitude"].idxmax(), "t2_ms"])
    t2_peak_ms = float(np.mean(list(t2_peaks.values())))
    peak_position = log_position(t2_peak_ms, T2_LIMITS_MS)
    radius_max_um = nice_log_upper(max(pore_radius.max(), throat_radius.max()))
    log_radius_min = (np.log10(radius_peak_um) - peak_position * np.log10(radius_max_um)) / (1.0 - peak_position)
    radius_limits_um = (float(10**log_radius_min), radius_max_um)
    alignment_radius_um = radius_peak_um if nmr_peak_radius_um is None else float(nmr_peak_radius_um)
    alignment_position = log_position(alignment_radius_um, radius_limits_um)
    if not 0.0 < alignment_position < 1.0:
        raise ValueError("NMR peak alignment radius must lie within the displayed CT radius axis")
    if nmr_peak_radius_um is None:
        t2_limits_ms = T2_LIMITS_MS
    else:
        log_t2_min = (
            np.log10(t2_peak_ms) - alignment_position * np.log10(T2_LIMITS_MS[1])
        ) / (1.0 - alignment_position)
        t2_limits_ms = (float(10**log_t2_min), T2_LIMITS_MS[1])

    assert np.isclose(pore_fraction.sum() + throat_fraction.sum(), 1.0)
    assert abs(log_position(alignment_radius_um, radius_limits_um) - log_position(t2_peak_ms, t2_limits_ms)) < 1e-12
    assert set(t2_peaks) == set(tests)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict] = []
    for name, values in (("pore body", pore_fraction), ("internal throat", throat_fraction)):
        for left, right, center, fraction in zip(edges[:-1], edges[1:], centers, values):
            source_rows.append(
                {
                    "domain": "PNM radius",
                    "series": name,
                    "test": "",
                    "x_left": left,
                    "x_right": right,
                    "x_value": center,
                    "x_unit": "um",
                    "fraction": fraction,
                }
            )
    for test, group in spectra.groupby("test", sort=False):
        signal = group["raw_spectrum_amplitude"].to_numpy(float)
        signal_fraction = signal / signal.sum()
        for t2_ms, fraction in zip(group["t2_ms"], signal_fraction):
            source_rows.append(
                {
                    "domain": "NNLS T2",
                    "series": "L-curve NNLS",
                    "test": test,
                    "x_left": np.nan,
                    "x_right": np.nan,
                    "x_value": t2_ms,
                    "x_unit": "ms",
                    "fraction": fraction,
                }
            )
    source_csv = output_dir / f"sample{sample_number}_nnls_pore_throat_source_data.csv"
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.4))
    ax_radius = fig.add_axes([0.11, 0.16, 0.76, 0.66])
    ax_t2 = fig.add_axes(ax_radius.get_position(), frameon=False)
    ax_t2.patch.set_visible(False)
    ax_radius.set_xscale("log")
    ax_t2.set_xscale("log")
    ax_radius.set_xlim(radius_limits_um)
    ax_t2.set_xlim(t2_limits_ms)
    ax_t2.xaxis.set_label_position("top")
    ax_t2.xaxis.tick_top()
    ax_t2.yaxis.set_label_position("right")
    ax_t2.yaxis.tick_right()
    ax_t2.spines["bottom"].set_visible(False)
    ax_t2.spines["left"].set_visible(False)
    ax_t2.spines["top"].set_visible(True)
    ax_t2.spines["right"].set_visible(True)

    overlap_fraction = np.minimum(pore_fraction, throat_fraction)
    assert np.all(overlap_fraction <= pore_fraction) and np.all(overlap_fraction <= throat_fraction)
    throat_handle = ax_radius.stairs(
        throat_fraction, edges, fill=True, edgecolor="black", facecolor="white", hatch="////", lw=1.0,
        label="Internal throat (hatch)",
    )
    overlap_handle = ax_radius.stairs(
        overlap_fraction, edges, fill=True, edgecolor="black", facecolor="#808080", lw=0.9,
        label="Histogram overlap",
    )
    pore_handle = ax_radius.stairs(
        pore_fraction, edges, fill=False, edgecolor="black", facecolor="none", lw=1.2,
        label="Pore body (outline)",
    )
    line_handles = []
    styles = {"first": "-", "second": "--"}
    for test, group in spectra.groupby("test", sort=False):
        signal_fraction = group["raw_spectrum_amplitude"].to_numpy(float)
        signal_fraction = signal_fraction / signal_fraction.sum()
        alpha = float(group["regularization"].iloc[0])
        line, = ax_t2.plot(
            group["t2_ms"], signal_fraction, color="black", ls=styles.get(test, "-"), lw=1.7,
            label=f"NNLS {test} (alpha={alpha:.3g})",
        )
        line_handles.append(line)

    ax_radius.axvline(alignment_radius_um, color="black", lw=0.9, ls=":", zorder=5)
    ax_radius.set_ylim(0, 1.13 * float(max(pore_fraction.max(), throat_fraction.max())))
    ax_t2.set_ylim(0, 1.16 * max(float(line.get_ydata().max()) for line in line_handles))
    ax_radius.set_xlabel(r"Pore-network radius, $r$ ($\mu$m)")
    ax_t2.set_xlabel(r"$T_2$ (ms)", labelpad=7)
    ax_radius.set_ylabel("CT/PNM volume fraction per log-radius bin")
    ax_t2.set_ylabel(r"NNLS signal fraction per log-$T_2$ bin")
    ax_radius.grid(axis="y", color="#D0D0D0", lw=0.6, alpha=0.8)
    ax_radius.tick_params(which="both", direction="out", length=3)
    ax_t2.tick_params(which="both", direction="out", length=3)
    ax_radius.legend(
        [pore_handle, throat_handle, overlap_handle, *line_handles],
        [
            pore_handle.get_label(), throat_handle.get_label(), overlap_handle.get_label(),
            *(line.get_label() for line in line_handles),
        ],
        loc="upper left", ncol=2, columnspacing=1.2, handlelength=2.3,
    )

    output_png = output_dir / f"sample{sample_number}_nnls_t2_pore_throat_volume_fraction.png"
    fig.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    extraction_parameters = None
    if network_extraction_summary is not None:
        extraction_run = json.loads(network_extraction_summary.read_text(encoding="utf-8"))
        prepare = extraction_run["prepare"]
        extraction_parameters = {
            key: prepare[key]
            for key in ("downsample", "shape_zyx", "source_voxel_size_um", "effective_voxel_size_um", "porosity")
        }

    manifest = {
        "sample": sample_id,
        "source_ct": None if source_ct is None else str(source_ct),
        "network_extraction_summary": (
            None if network_extraction_summary is None else str(network_extraction_summary)
        ),
        "network_extraction_parameters": extraction_parameters,
        "network_dir": str(network_dir),
        "output_png": str(output_png),
        "source_data_csv": str(source_csv),
        "nmr_method": "L-curve NNLS",
        "nmr_input": nmr_provenance,
        "nmr_tests": {
            test: {
                "regularization": float(group["regularization"].iloc[0]),
                "t2_peak_ms": t2_peaks[test],
                "normalization": "raw NNLS amplitude divided by the sum within that test",
            }
            for test, group in spectra.groupby("test", sort=False)
        },
        "network": {
            "pore_count": int(len(pore_radius)),
            "internal_throat_count_total": int(len(throats)),
            "internal_throat_count_used": int(len(throat_radius)),
            "excluded_boundary_throat_count": int(len(throats_all) - len(throats)),
            "excluded_nonpositive_or_nonfinite_internal_throat_count": int(len(throats) - len(throat_radius)),
            "shared_denominator_m3": shared_network_volume,
            "pore_body_total_fraction": float(pore_fraction.sum()),
            "internal_throat_total_fraction": float(throat_fraction.sum()),
            "binning": f"{N_BINS} shared logarithmic radius bins",
        },
        "peak_alignment": {
            "network_target": (
                "combined pore-body plus internal-throat volume-fraction peak"
                if nmr_peak_radius_um is None else "user-specified CT radius display position"
            ),
            "combined_network_peak_um": radius_peak_um,
            "radius_peak_um": radius_peak_um,
            "alignment_radius_um": alignment_radius_um,
            "t2_peak_ms": t2_peak_ms,
            "radius_axis_um": list(radius_limits_um),
            "t2_axis_ms": list(t2_limits_ms),
            "rule": "display-axis alignment only via T2-axis limits; original radius and T2 values are unchanged",
        },
        "presentation": {
            "palette": "black, white, and gray only",
            "left_y_axis": "CT/PNM volume fraction per logarithmic radius bin",
            "right_y_axis": "sum-normalized NNLS signal fraction per logarithmic T2 bin",
            "peak_marker": "black dotted vertical line",
            "pore_body_style": "white interior with black outline only",
            "internal_throat_style": "white interior with black diagonal hatch",
            "histogram_overlap_style": "solid gray fill using min(pore_fraction, throat_fraction) per bin",
            "histogram_overlap_interpretation": "display-only vertical overlap, not physical shared pore/throat volume",
            "in_plot_annotation": "none; legend only",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", type=int, default=[24])
    parser.add_argument("--tests", nargs="+", choices=("first", "second"), default=["first", "second"])
    parser.add_argument("--network-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-ct")
    parser.add_argument("--network-extraction-summary")
    parser.add_argument("--nmr-peak-radius-um", type=float)
    parser.add_argument("--nmr-spectrum")
    parser.add_argument("--nmr-summary")
    parser.add_argument("--nmr-spectrum-table")
    parser.add_argument("--nmr-parameters-table")
    args = parser.parse_args()
    if len(args.samples) != 1 and any((args.network_dir, args.output_dir, args.source_ct, args.network_extraction_summary)):
        parser.error("custom paths require exactly one sample")
    for sample_number in args.samples:
        render_sample(
            sample_number,
            tests=tuple(args.tests),
            network_dir_override=None if args.network_dir is None else Path(args.network_dir),
            output_dir_override=None if args.output_dir is None else Path(args.output_dir),
            source_ct=None if args.source_ct is None else Path(args.source_ct),
            network_extraction_summary=(
                None if args.network_extraction_summary is None else Path(args.network_extraction_summary)
            ),
            nmr_peak_radius_um=args.nmr_peak_radius_um,
            nmr_spectrum=None if args.nmr_spectrum is None else Path(args.nmr_spectrum),
            nmr_summary=None if args.nmr_summary is None else Path(args.nmr_summary),
            nmr_spectrum_table=(
                None if args.nmr_spectrum_table is None else Path(args.nmr_spectrum_table)
            ),
            nmr_parameters_table=(
                None if args.nmr_parameters_table is None else Path(args.nmr_parameters_table)
            ),
        )


if __name__ == "__main__":
    main()
