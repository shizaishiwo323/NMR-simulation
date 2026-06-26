"""Extract sample 16 pore sizes with pnextract and overlay them with T2 spectra."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


DEFAULT_TIFF = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\data_inventory"
    r"\ct_backed_samples_raw_copy_20260605\sample_16_Grainstone"
    r"\CT_slices\1-CTseg\5-16seged.tiff"
)
DEFAULT_PNEXTRACT_EXE = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\code\vendor\pnextract\bin\pnextract.exe"
)
DEFAULT_EXPERIMENT_SPECTRUM = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\data_inventory"
    r"\ct_backed_samples_raw_copy_20260605\sample_16_Grainstone"
    r"\NMR_original\16-1\T2CPMG\Spectra\spectrum.csv"
)
DEFAULT_SIM_SPECTRUM = Path(
    r"C:\Users\imgw\Documents\Codex\NMR模拟"
    r"\simulation_outputs\sample_16_random10_nmr_px1p92_b5"
    r"\sample16_average_10_slices_t2_inversion.csv"
)
DEFAULT_SIM_3D_SPECTRUM = Path(
    r"C:\Users\imgw\Documents\Codex\NMR模拟"
    r"\simulation_outputs\sample_16_rev336_pygimli_native_3d_down4_full"
    r"\pygimli_tetra_nmr_t2_inversion.csv"
)


def safe_console_text(value: object) -> str:
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def t2_ms_to_pore_diameter_um(t2_ms: float | np.ndarray, rho_um_per_ms: float) -> float | np.ndarray:
    return 2.0 * rho_um_per_ms * np.asarray(t2_ms)


def pore_diameter_um_to_t2_ms(diameter_um: float | np.ndarray, rho_um_per_ms: float) -> float | np.ndarray:
    return np.asarray(diameter_um) / (2.0 * rho_um_per_ms)


def display_alignment_scale(
    histogram_centers_t2_ms: np.ndarray,
    histogram_values: np.ndarray,
    target_peak_t2_ms: float,
) -> tuple[float, float]:
    source_peak_t2_ms = float(histogram_centers_t2_ms[int(np.argmax(histogram_values))])
    if source_peak_t2_ms <= 0:
        raise ValueError("Histogram peak T2 must be positive for display alignment.")
    return float(target_peak_t2_ms / source_peak_t2_ms), source_peak_t2_ms


def display_alignment_shift(
    histogram_centers_t2_ms: np.ndarray,
    histogram_values: np.ndarray,
    target_peak_t2_ms: float,
) -> tuple[float, float]:
    source_peak_t2_ms = float(histogram_centers_t2_ms[int(np.argmax(histogram_values))])
    return float(target_peak_t2_ms - source_peak_t2_ms), source_peak_t2_ms


def aligned_log_axis_min_for_peak(
    bottom_xlim: tuple[float, float],
    bottom_peak: float,
    top_peak: float,
    top_max: float,
) -> float:
    bottom_min, bottom_max = bottom_xlim
    bottom_fraction = (np.log10(bottom_peak) - np.log10(bottom_min)) / (
        np.log10(bottom_max) - np.log10(bottom_min)
    )
    if not 0 < bottom_fraction < 1:
        raise ValueError("Bottom peak must be inside the bottom axis range.")
    top_log_min = (np.log10(top_peak) - bottom_fraction * np.log10(top_max)) / (1.0 - bottom_fraction)
    return float(10**top_log_min)


def spectrum_peak_t2_ms(spectrum: pd.DataFrame) -> float:
    if spectrum.empty:
        raise ValueError("Cannot find peak of an empty spectrum.")
    return float(spectrum.loc[spectrum["normalized_amplitude"].idxmax(), "t2_ms"])


def load_node2_pore_table(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["pore_id", "pore_volume_m3", "pore_radius_m", "shape_factor", "clay_volume"])
    raw = raw[np.isfinite(raw["pore_radius_m"]) & (raw["pore_radius_m"] > 0)].copy()
    raw["pore_volume_um3"] = raw["pore_volume_m3"] * 1e18
    raw["pore_radius_um"] = raw["pore_radius_m"] * 1e6
    raw["pore_diameter_um"] = 2.0 * raw["pore_radius_um"]
    return raw


def write_pnextract_mhd(mhd_path: Path, raw_path: Path, dims_xyz: tuple[int, int, int], voxel_size_um: float) -> None:
    x_size, y_size, z_size = dims_xyz
    text = "\n".join(
        [
            "ObjectType =  Image",
            "NDims =       3",
            "ElementType = MET_UCHAR",
            "ElementByteOrderMSB = False",
            f"DimSize =     {x_size} {y_size} {z_size}",
            f"ElementSize =  {voxel_size_um:g} {voxel_size_um:g} {voxel_size_um:g}",
            "Offset =       0 0 0",
            f"ElementDataFile = {raw_path.name}",
            "threshold 0 0",
            "write_elements false",
            "write_cnm true",
            "overwrite true",
            "",
        ]
    )
    mhd_path.write_text(text, encoding="utf-8")


def write_pnextract_raw_from_tiff(
    tiff_path: Path,
    raw_path: Path,
    pore_value: int,
    solid_value: int,
) -> tuple[int, int, int, dict[str, int]]:
    counts = {"pore_voxels": 0, "solid_voxels": 0}
    with tifffile.TiffFile(str(tiff_path)) as tif, raw_path.open("wb") as out:
        z_size = len(tif.pages)
        first_shape = tif.pages[0].shape
        y_size, x_size = int(first_shape[0]), int(first_shape[1])
        for page in tif.pages:
            arr = page.asarray()
            if arr.shape != first_shape:
                raise ValueError(f"TIFF pages have inconsistent shapes: {arr.shape} vs {first_shape}")
            pore = arr == pore_value
            solid = arr == solid_value
            if np.any(~(pore | solid)):
                unexpected = np.unique(arr[~(pore | solid)])
                raise ValueError(f"Unexpected labels in TIFF: {unexpected[:20].tolist()}")
            counts["pore_voxels"] += int(np.sum(pore))
            counts["solid_voxels"] += int(np.sum(solid))
            binary = np.where(pore, 0, 1).astype(np.uint8)
            out.write(binary.tobytes(order="C"))
    return x_size, y_size, z_size, counts


def load_spectrum(path: Path, name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if {"t2_ms", "amplitude"}.issubset(raw.columns):
        frame = raw[["t2_ms", "amplitude"]].copy()
    else:
        raw = pd.read_csv(path, header=None)
        frame = raw.iloc[:, :2].copy()
        frame.columns = ["t2_ms", "amplitude"]
    frame["series"] = name
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[(frame["t2_ms"] > 0) & (frame["amplitude"] >= 0)].copy()
    frame["normalized_amplitude"] = frame["amplitude"] / max(float(frame["amplitude"].max()), 1e-30)
    return frame


def run_pnextract(pnextract_exe: Path, mhd_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(pnextract_exe), mhd_path.name],
        cwd=str(output_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def save_overlay_plot(
    pores: pd.DataFrame,
    experiment: pd.DataFrame,
    simulation: pd.DataFrame,
    output_path: Path,
    rho_um_per_ms: float,
    bins: int,
    align_peaks_for_display: bool = False,
    display_alignment_mode: str = "scale",
    histogram_axis_mode: str = "converted_t2",
    xlim_min_ms: float | None = None,
    xlim_max_ms: float | None = None,
    top_axis_min_um: float | None = None,
    top_axis_max_um: float | None = None,
    additional_spectra: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    pores = pores.copy()
    pores["equivalent_t2_ms"] = pore_diameter_um_to_t2_ms(pores["pore_diameter_um"].to_numpy(float), rho_um_per_ms)
    t2_min = 1e-2
    t2_max = max(1e5, float(pores["equivalent_t2_ms"].max()) * 1.2)
    if histogram_axis_mode == "top_pore_diameter":
        diameter_edges = np.logspace(
            np.log10(max(float(pores["pore_diameter_um"].min()) * 0.8, 1e-9)),
            np.log10(float(pores["pore_diameter_um"].max()) * 1.2),
            bins + 1,
        )
        hist, diameter_edges = np.histogram(
            pores["pore_diameter_um"],
            bins=diameter_edges,
            weights=pores["pore_volume_um3"],
        )
        centers = np.sqrt(diameter_edges[:-1] * diameter_edges[1:])
        edges = pore_diameter_um_to_t2_ms(diameter_edges, rho_um_per_ms)
    else:
        edges = np.logspace(np.log10(t2_min), np.log10(t2_max), bins + 1)
        hist, edges = np.histogram(
            pores["equivalent_t2_ms"],
            bins=edges,
            weights=pores["pore_volume_um3"],
        )
        centers = np.sqrt(edges[:-1] * edges[1:])
    hist_norm = hist / max(float(hist.max()), 1e-30)
    hist_frame = pd.DataFrame(
        {
            "equivalent_t2_ms_center": pore_diameter_um_to_t2_ms(centers, rho_um_per_ms)
            if histogram_axis_mode == "top_pore_diameter"
            else centers,
            "pore_diameter_um_center": centers
            if histogram_axis_mode == "top_pore_diameter"
            else t2_ms_to_pore_diameter_um(centers, rho_um_per_ms),
            "volume_weighted_count": hist,
            "normalized_volume_weighted_count": hist_norm,
        }
    )
    alignment_scale = 1.0
    alignment_shift_ms = 0.0
    target_peak_t2_ms = np.nan
    source_peak_t2_ms = np.nan
    display_edges = edges
    display_hist_norm = hist_norm
    if align_peaks_for_display:
        target_peak_t2_ms = spectrum_peak_t2_ms(experiment)
        if display_alignment_mode == "shift":
            alignment_shift_ms, source_peak_t2_ms = display_alignment_shift(centers, hist_norm, target_peak_t2_ms)
            display_edges = edges + alignment_shift_ms
            valid = (display_edges[:-1] > 0) & (display_edges[1:] > 0)
            display_edges = display_edges[np.r_[valid, False] | np.r_[False, valid]]
            display_hist_norm = hist_norm[valid]
        else:
            alignment_scale, source_peak_t2_ms = display_alignment_scale(centers, hist_norm, target_peak_t2_ms)
            display_edges = edges * alignment_scale
    if histogram_axis_mode == "top_pore_diameter":
        hist_frame["display_equivalent_t2_ms_center"] = hist_frame["equivalent_t2_ms_center"]
    else:
        hist_frame["display_equivalent_t2_ms_center"] = centers * alignment_scale + alignment_shift_ms
    hist_frame["display_alignment_scale"] = alignment_scale
    hist_frame["display_alignment_shift_ms"] = alignment_shift_ms

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    hist_label = "pnextract pore diameter histogram"
    if align_peaks_for_display and histogram_axis_mode != "top_pore_diameter":
        hist_label += f" (display-{display_alignment_mode})"
    top_ax = None
    if histogram_axis_mode == "top_pore_diameter":
        top_ax = ax.twiny()
        top_ax.set_xscale("log")
        top_ax.stairs(hist_norm, diameter_edges, color="#5b6f95", linewidth=2.0, label=hist_label)
    else:
        ax.stairs(display_hist_norm, display_edges, color="#5b6f95", linewidth=2.0, label=hist_label)
    ax.plot(
        experiment["t2_ms"],
        experiment["normalized_amplitude"],
        color="#0b6e69",
        linewidth=2.2,
        label="experimental T2 inversion",
    )
    ax.plot(
        simulation["t2_ms"],
        simulation["normalized_amplitude"],
        color="#1f2937",
        linewidth=2.2,
        label=str(simulation["series"].iloc[0]) if "series" in simulation.columns and not simulation.empty else "2D simulation T2 inversion",
    )
    extra_styles = [
        {"color": "#b45309", "linestyle": "--", "linewidth": 2.2},
        {"color": "#7c3aed", "linestyle": "-.", "linewidth": 2.0},
    ]
    for i, spectrum in enumerate(additional_spectra or []):
        label = str(spectrum["series"].iloc[0]) if "series" in spectrum.columns and not spectrum.empty else f"additional T2 {i + 1}"
        style = extra_styles[i % len(extra_styles)]
        ax.plot(
            spectrum["t2_ms"],
            spectrum["normalized_amplitude"],
            label=label,
            **style,
        )
    ax.set_xscale("log")
    bottom_xlim = (xlim_min_ms if xlim_min_ms is not None else t2_min, xlim_max_ms if xlim_max_ms is not None else t2_max)
    ax.set_xlim(*bottom_xlim)
    ax.set_ylim(bottom=-0.02, top=1.08)
    if histogram_axis_mode == "top_pore_diameter":
        ax.set_xlabel("T2 (ms)")
    elif align_peaks_for_display:
        if display_alignment_mode == "shift":
            ax.set_xlabel(
                "display T2 (ms); pnextract histogram shifted "
                f"by {alignment_shift_ms:.0f} ms to align {source_peak_t2_ms:.0f} ms with {target_peak_t2_ms:.0f} ms"
            )
        else:
            ax.set_xlabel(
                "display T2 (ms); pnextract histogram scaled "
                f"by {alignment_scale:.3g} to align {source_peak_t2_ms:.0f} ms with {target_peak_t2_ms:.0f} ms"
            )
    else:
        ax.set_xlabel(f"T2 (ms), pore diameter converted by d = 2 rho2 T2, rho2={rho_um_per_ms:g} um/ms")
    ax.set_ylabel("normalized amplitude / volume-weighted frequency")
    ax.set_title("Sample 16 pore-size distribution vs T2 spectra")
    ax.grid(alpha=0.25, which="both")
    if top_ax is not None:
        target_peak_t2_ms = spectrum_peak_t2_ms(experiment)
        top_peak_um = float(centers[int(np.argmax(hist_norm))])
        top_max_um = top_axis_max_um if top_axis_max_um is not None else t2_ms_to_pore_diameter_um(bottom_xlim[1], rho_um_per_ms)
        top_min_um = (
            top_axis_min_um
            if top_axis_min_um is not None
            else aligned_log_axis_min_for_peak(bottom_xlim, target_peak_t2_ms, top_peak_um, top_max_um)
        )
        top_ax.set_xlim(top_min_um, top_max_um)
        top_ax.set_xlabel("pore diameter (um)")
        lines, labels = ax.get_legend_handles_labels()
        top_lines, top_labels = top_ax.get_legend_handles_labels()
        ax.legend(top_lines + lines, top_labels + labels, loc="best")
        hist_frame["top_axis_min_um"] = top_min_um
        hist_frame["top_axis_max_um"] = top_max_um
    else:
        ax.legend(loc="best")
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda t: t2_ms_to_pore_diameter_um((np.asarray(t) - alignment_shift_ms) / alignment_scale, rho_um_per_ms),
                lambda d: pore_diameter_um_to_t2_ms(d, rho_um_per_ms) * alignment_scale + alignment_shift_ms,
            ),
        )
        secax.set_xlabel("equivalent pore diameter (um; before display alignment)" if align_peaks_for_display else "equivalent pore diameter (um)")
        hist_frame["top_axis_min_um"] = np.nan
        hist_frame["top_axis_max_um"] = np.nan
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return hist_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--pnextract-exe", type=Path, default=DEFAULT_PNEXTRACT_EXE)
    parser.add_argument("--experiment-spectrum", type=Path, default=DEFAULT_EXPERIMENT_SPECTRUM)
    parser.add_argument("--simulation-spectrum", type=Path, default=DEFAULT_SIM_SPECTRUM)
    parser.add_argument("--simulation-3d-spectrum", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs/sample_16_pnextract_t2_overlay"))
    parser.add_argument("--pore-value", type=int, default=2)
    parser.add_argument("--solid-value", type=int, default=1)
    parser.add_argument("--voxel-size-um", type=float, default=1.92)
    parser.add_argument("--rho-um-per-ms", type=float, default=0.015)
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument("--skip-pnextract", action="store_true")
    parser.add_argument("--align-peaks-for-display", action="store_true")
    parser.add_argument("--display-alignment-mode", choices=["scale", "shift"], default="scale")
    parser.add_argument("--histogram-axis-mode", choices=["converted_t2", "top_pore_diameter"], default="converted_t2")
    parser.add_argument("--xlim-min-ms", type=float, default=None)
    parser.add_argument("--xlim-max-ms", type=float, default=None)
    parser.add_argument("--top-axis-min-um", type=float, default=None)
    parser.add_argument("--top-axis-max-um", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sample16_pnextract_binary.raw"
    mhd_path = args.output_dir / "sample16_pnextract_binary.mhd"
    node2_path = args.output_dir / "sample16_pnextract_binary_node2.dat"
    log_path = args.output_dir / "pnextract_stdout.log"

    if not args.skip_pnextract:
        dims = write_pnextract_raw_from_tiff(args.input_tiff, raw_path, args.pore_value, args.solid_value)
        x_size, y_size, z_size, counts = dims
        write_pnextract_mhd(mhd_path, raw_path, (x_size, y_size, z_size), args.voxel_size_um)
        completed = run_pnextract(args.pnextract_exe, mhd_path, args.output_dir)
        log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    else:
        if not node2_path.exists():
            raise FileNotFoundError(f"--skip-pnextract requires existing {node2_path}")
        with tifffile.TiffFile(str(args.input_tiff)) as tif:
            y_size, x_size = tif.pages[0].shape
            z_size = len(tif.pages)
        old_manifest_path = args.output_dir / "run_manifest.json"
        if old_manifest_path.exists():
            old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
            counts = old_manifest.get("voxel_counts", {"pore_voxels": -1, "solid_voxels": -1})
        else:
            counts = {"pore_voxels": -1, "solid_voxels": -1}

    pores = load_node2_pore_table(node2_path)
    pores["equivalent_t2_ms"] = pore_diameter_um_to_t2_ms(pores["pore_diameter_um"].to_numpy(float), args.rho_um_per_ms)
    pores_csv = args.output_dir / "pnextract_pore_radius_table.csv"
    pores.to_csv(pores_csv, index=False)

    experiment = load_spectrum(args.experiment_spectrum, "experimental T2 inversion")
    simulation = load_spectrum(args.simulation_spectrum, "2D simulation T2 inversion")
    additional_spectra = []
    if args.simulation_3d_spectrum is not None:
        simulation_3d = load_spectrum(args.simulation_3d_spectrum, "3D pyGIMLi T2 inversion")
        additional_spectra.append(simulation_3d)
    figure_path = args.output_dir / "sample16_pnextract_pore_histogram_vs_t2.png"
    hist_frame = save_overlay_plot(
        pores,
        experiment,
        simulation,
        figure_path,
        args.rho_um_per_ms,
        args.histogram_bins,
        align_peaks_for_display=args.align_peaks_for_display,
        display_alignment_mode=args.display_alignment_mode,
        histogram_axis_mode=args.histogram_axis_mode,
        xlim_min_ms=args.xlim_min_ms,
        xlim_max_ms=args.xlim_max_ms,
        top_axis_min_um=args.top_axis_min_um,
        top_axis_max_um=args.top_axis_max_um,
        additional_spectra=additional_spectra,
    )
    hist_csv = args.output_dir / "pnextract_pore_histogram_equivalent_t2.csv"
    hist_frame.to_csv(hist_csv, index=False)

    manifest = {
        "input_tiff": str(args.input_tiff.resolve()),
        "pnextract_exe": str(args.pnextract_exe.resolve()),
        "mhd_path": str(mhd_path.resolve()),
        "raw_path": str(raw_path.resolve()),
        "node2_path": str(node2_path.resolve()),
        "experiment_spectrum": str(args.experiment_spectrum.resolve()),
        "simulation_spectrum": str(args.simulation_spectrum.resolve()),
        "simulation_3d_spectrum": str(args.simulation_3d_spectrum.resolve()) if args.simulation_3d_spectrum else None,
        "output_figure": str(figure_path.resolve()),
        "pore_table_csv": str(pores_csv.resolve()),
        "histogram_csv": str(hist_csv.resolve()),
        "label_convention": {
            str(args.pore_value): "pore space mapped to 0 for pnextract",
            str(args.solid_value): "solid matrix mapped to 1",
        },
        "voxel_size_um": args.voxel_size_um,
        "dims_xyz": [int(x_size), int(y_size), int(z_size)],
        "voxel_counts": counts,
        "pore_count_from_node2": int(len(pores)),
        "rho_um_per_ms": args.rho_um_per_ms,
        "conversion": {
            "formula": "1/T2 ~= rho2*S/V; using the cited cylindrical-pore approximation r_um=rho_um_per_ms*T2_ms, so d_um=2*rho_um_per_ms*T2_ms",
            "assumption": "fast-diffusion, surface-relaxation-dominated, cylindrical equivalent pore diameter",
        },
        "display": {
            "align_peaks_for_display": bool(args.align_peaks_for_display),
            "display_alignment_mode": args.display_alignment_mode,
            "histogram_axis_mode": args.histogram_axis_mode,
            "histogram_display_alignment_scale": float(hist_frame["display_alignment_scale"].iloc[0]),
            "histogram_display_alignment_shift_ms": float(hist_frame["display_alignment_shift_ms"].iloc[0]),
            "xlim_min_ms": args.xlim_min_ms,
            "xlim_max_ms": args.xlim_max_ms,
            "top_axis_min_um": float(hist_frame["top_axis_min_um"].iloc[0]),
            "top_axis_max_um": float(hist_frame["top_axis_max_um"].iloc[0]),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pores parsed: {len(pores)}")
    print(f"Figure: {safe_console_text(figure_path.resolve())}")


if __name__ == "__main__":
    main()
