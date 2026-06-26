from pathlib import Path

import numpy as np
import pandas as pd

from advanced_tools.sample16_pnextract_t2_overlay import (
    aligned_log_axis_min_for_peak,
    display_alignment_scale,
    display_alignment_shift,
    load_node2_pore_table,
    parse_args,
    pore_diameter_um_to_t2_ms,
    safe_console_text,
    save_overlay_plot,
    t2_ms_to_pore_diameter_um,
    write_pnextract_mhd,
)


def test_node2_parser_reads_pore_radius_and_volume(tmp_path: Path):
    path = tmp_path / "sample_node2.dat"
    path.write_text(
        "     1 8.000000E-18 1.000000E-06 2.500000E-02 0.000000E+00\n"
        "     2 2.700000E-17 1.500000E-06 3.000000E-02 0.000000E+00\n",
        encoding="utf-8",
    )

    pores = load_node2_pore_table(path)

    assert pores["pore_id"].tolist() == [1, 2]
    assert np.allclose(pores["pore_radius_um"], [1.0, 1.5])
    assert np.allclose(pores["pore_diameter_um"], [2.0, 3.0])
    assert np.allclose(pores["pore_volume_um3"], [8.0, 27.0])


def test_t2_pore_diameter_conversion_uses_cylindrical_fast_diffusion():
    rho = 0.005

    assert t2_ms_to_pore_diameter_um(1000.0, rho) == 10.0
    assert pore_diameter_um_to_t2_ms(10.0, rho) == 1000.0


def test_display_alignment_scale_maps_histogram_peak_to_t2_peak():
    histogram_centers = np.array([100.0, 1000.0, 10000.0])
    histogram_values = np.array([0.1, 0.2, 1.0])
    target_peak_t2_ms = 1500.0

    scale, source_peak = display_alignment_scale(histogram_centers, histogram_values, target_peak_t2_ms)

    assert source_peak == 10000.0
    assert scale == 0.15


def test_display_alignment_shift_translates_histogram_peak_to_t2_peak():
    histogram_centers = np.array([100.0, 1000.0, 10000.0])
    histogram_values = np.array([0.1, 0.2, 1.0])
    target_peak_t2_ms = 1500.0

    shift, source_peak = display_alignment_shift(histogram_centers, histogram_values, target_peak_t2_ms)

    assert source_peak == 10000.0
    assert shift == -8500.0


def test_aligned_log_axis_min_places_top_peak_at_bottom_peak_position():
    bottom_xlim = (1e-2, 1e5)
    bottom_peak = 1_657.67
    top_peak = 361.7189443541622
    top_max = 3000.0

    top_min = aligned_log_axis_min_for_peak(bottom_xlim, bottom_peak, top_peak, top_max)

    bottom_fraction = (np.log10(bottom_peak) - np.log10(bottom_xlim[0])) / (
        np.log10(bottom_xlim[1]) - np.log10(bottom_xlim[0])
    )
    top_fraction = (np.log10(top_peak) - np.log10(top_min)) / (np.log10(top_max) - np.log10(top_min))
    assert np.isclose(top_fraction, bottom_fraction)


def test_mhd_writer_records_dimensions_spacing_and_threshold(tmp_path: Path):
    raw_path = tmp_path / "image.raw"
    raw_path.write_bytes(b"")
    mhd_path = tmp_path / "image.mhd"

    write_pnextract_mhd(
        mhd_path=mhd_path,
        raw_path=raw_path,
        dims_xyz=(4, 5, 6),
        voxel_size_um=1.92,
    )

    text = mhd_path.read_text(encoding="utf-8")
    assert "DimSize =     4 5 6" in text
    assert "ElementSize =  1.92 1.92 1.92" in text
    assert "ElementDataFile = image.raw" in text
    assert "threshold 0 0" in text
    assert "write_elements false" in text


def test_overlay_plot_accepts_additional_3d_spectrum(tmp_path: Path):
    pores = pd.DataFrame(
        {
            "pore_diameter_um": [1.0, 2.0, 4.0],
            "pore_volume_um3": [1.0, 3.0, 2.0],
        }
    )
    experiment = pd.DataFrame(
        {
            "t2_ms": [1.0, 10.0, 100.0],
            "normalized_amplitude": [0.1, 1.0, 0.2],
        }
    )
    simulation = pd.DataFrame(
        {
            "t2_ms": [1.0, 10.0, 100.0],
            "normalized_amplitude": [0.2, 0.8, 0.3],
        }
    )
    simulation_3d = pd.DataFrame(
        {
            "t2_ms": [1.0, 10.0, 100.0],
            "normalized_amplitude": [0.3, 0.6, 1.0],
            "series": ["3D pyGIMLi T2 inversion"] * 3,
        }
    )
    output_path = tmp_path / "overlay.png"

    hist = save_overlay_plot(
        pores,
        experiment,
        simulation,
        output_path,
        rho_um_per_ms=0.015,
        bins=4,
        additional_spectra=[simulation_3d],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert "equivalent_t2_ms_center" in hist.columns


def test_generic_dual_axis_overlay_entrypoint_reuses_sample16_workflow():
    from advanced_tools.pnextract_t2_dual_axis_overlay import main

    assert callable(main)


def test_cli_defaults_do_not_show_3d_spectrum(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sample16_pnextract_t2_overlay.py"])

    args = parse_args()

    assert args.simulation_3d_spectrum is None


def test_safe_console_text_escapes_unicode_for_legacy_stdout(monkeypatch):
    class LegacyStdout:
        encoding = "cp1252"

    monkeypatch.setattr("sys.stdout", LegacyStdout())

    text = safe_console_text("NMR模拟")

    assert text == "NMR\\u6a21\\u62df"
