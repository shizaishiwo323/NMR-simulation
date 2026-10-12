"""Read-only numerical diagnosis of Sample 24 CT versus grouped NMR.

Run with conda base Python. Write diagnostics to a new analysis directory.
Sphere/cylinder conversions and voxel-face S/V are conditional diagnostics,
not replacements for the actual finite-element simulation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile

SOURCE = Path("simulation_outputs/sample_24_grouped_nmr_lcurve_20261011")
BRD = Path("simulation_outputs/sample_24_grouped_nmr_brd_alpha0p1_20261011")
OUTPUT = Path("simulation_outputs/sample_24_ct_nmr_diagnosis_20261011")


def main():
    OUTPUT.mkdir(exist_ok=True)
    manifest = json.loads((BRD / "run_manifest.json").read_text(encoding="utf-8"))
    overlay = pd.read_csv(BRD / "overlay_source_data.csv")
    spectrum = pd.read_csv(BRD / "simulation_t2_brd_alpha0p1.csv")
    groups = pd.read_csv(SOURCE / "porosity_groups_and_representatives.csv")
    mesh = pd.read_csv(SOURCE / "representative_mesh_summary.csv")
    profile = pd.read_csv(SOURCE / "slice_porosity_profile.csv")
    decay = pd.read_csv(SOURCE / "group_weighted_normalized_decay.csv")
    voxel = manifest["parameters"]["pixel_size_um"]
    bulk = manifest["parameters"]["bulk_t2_ms"]
    rho = manifest["parameters"]["rho_solid_um_per_ms"]
    rows = groups.merge(mesh, on="group_id")
    rows["perimeter_over_area_um_inv"] = rows.solid_liquid_boundary_length_um / rows.mesh_area_um2
    rows["fast_diffusion_mean_rate_t2_ms"] = 1 / (1 / bulk + rho * rows.perimeter_over_area_um_inv)
    representative_phi = profile.porosity.to_numpy()[rows.used_representative_index]
    rows["raster_pore_area_um2"] = representative_phi * 822 * 822 * voxel ** 2
    rows["mesh_to_raster_pore_area_ratio"] = rows.mesh_area_um2 / rows.raster_pore_area_um2
    count_weight = rows.weight_slice_count.to_numpy(float)
    volume_weight = count_weight * rows.mesh_area_um2.to_numpy(float)
    decay_matrix = np.column_stack([pd.read_csv(path).average_normalized_signal for path in rows.decay_csv])
    count_signal = decay_matrix @ (count_weight / count_weight.sum())
    volume_signal = decay_matrix @ (volume_weight / volume_weight.sum())
    assert np.allclose(count_signal, decay.group_weighted_normalized_signal)
    pd.DataFrame({"time_ms": decay.time_ms, "original_count_weighted_signal": count_signal,
                  "diagnostic_mesh_volume_weighted_signal": volume_signal}).to_csv(OUTPUT / "weighting_diagnostic_decay.csv", index=False)
    rows.to_csv(OUTPUT / "representative_geometry_diagnostics.csv", index=False)
    ct_peaks = {}
    for series in ["CT pore body", "CT internal throat"]:
        frame = overlay[overlay.series == series]
        peak = frame.loc[frame.fraction.idxmax()]
        radius = float(peak.radius_um)
        ct_peaks[series] = {"peak_bin_center_radius_um": radius,
                            "peak_bin_edges_um": [float(peak.radius_left_um), float(peak.radius_right_um)],
                            "sphere_t2_ms": float(1 / (1 / bulk + 3 * rho / radius)),
                            "cylinder_t2_ms": float(1 / (1 / bulk + 2 * rho / radius))}
    peak_ms = manifest["peak_t2_ms"]
    peak_mapping = {"t2_ms": peak_ms,
                    "existing_display_radius_um": 0.003 * manifest["reference_display_mapping"]["rho2_effective_um_per_s"] * peak_ms,
                    "no_bulk_sphere_radius_um": 3 * rho * peak_ms,
                    "bulk_corrected_sphere_radius_um": 3 * rho / (1 / peak_ms - 1 / bulk),
                    "bulk_corrected_cylinder_radius_um": 2 * rho / (1 / peak_ms - 1 / bulk),
                    "bulk_rate_fraction": peak_ms / bulk}
    # Cubic-voxel exposed-face area: internal pore-solid faces only, ROI cuts excluded.
    # This estimator has staircase surface bias and is not mesh-equivalent S/V.
    faces = np.zeros(3, dtype=np.int64)
    pore_count = 0
    previous = None
    with tifffile.TiffFile(manifest["input_tiff"]) as tif:
        for page in tif.pages:
            water = page.asarray() == 0
            pore_count += np.count_nonzero(water)
            faces[0] += np.count_nonzero(water[:, 1:] != water[:, :-1])
            faces[1] += np.count_nonzero(water[1:, :] != water[:-1, :])
            if previous is not None:
                faces[2] += np.count_nonzero(water != previous)
            previous = water
    assert pore_count == 63744276
    sv3d = float(faces.sum() / (pore_count * voxel))
    pa2d = float(np.average(rows.perimeter_over_area_um_inv, weights=count_weight))
    valid_t2 = spectrum.t2_ms < bulk
    radius_sphere = np.full(len(spectrum), np.nan)
    radius_sphere[valid_t2] = 3 * rho / (1 / spectrum.t2_ms[valid_t2] - 1 / bulk)
    pd.DataFrame({"t2_ms": spectrum.t2_ms, "signal_fraction": spectrum.signal_fraction,
                  "conditional_bulk_corrected_sphere_radius_um": radius_sphere}).to_csv(OUTPUT / "conditional_radius_mapping.csv", index=False)
    metrics = {
        "scope": "Diagnostic calculations on existing results; no new NMR solve or fitted physical parameters",
        "ct_peaks": ct_peaks, "brd_peak_mapping": peak_mapping,
        "lcurve_peak_t2_ms": json.loads((SOURCE / "run_manifest.json").read_text(encoding="utf-8"))["simulation_t2_peak_ms"],
        "group_porosity_range": [float(profile.porosity.min()), float(profile.porosity.max())],
        "representative_mesh_area_ratio_range": [float(rows.mesh_to_raster_pore_area_ratio.min()), float(rows.mesh_to_raster_pore_area_ratio.max())],
        "count_weighted_mesh_perimeter_over_area_um_inv": pa2d,
        "count_weighted_initial_fast_diffusion_t2_ms": 1 / (1 / bulk + rho * pa2d),
        "first_step_apparent_t2_ms": float(-5 / np.log(count_signal[1] / count_signal[0])),
        "voxel_internal_faces_xyz": faces.tolist(), "voxel_3d_surface_over_volume_um_inv": sv3d,
        "voxel_3d_initial_mean_rate_t2_ms": 1 / (1 / bulk + rho * sv3d),
        "voxel_3d_surface_estimator_caveat": "Axis-aligned exposed faces with staircase bias; ROI exterior excluded; not a full 3D solve or a converged area estimator",
        "count_vs_mesh_volume_weighted_decay_relative_difference": float(np.linalg.norm(count_signal - volume_signal) / np.linalg.norm(count_signal)),
        "count_vs_mesh_volume_weighted_decay_max_absolute_difference": float(np.max(np.abs(count_signal - volume_signal))),
        "spectral_mass_above_bulk_t2": float(spectrum.loc[~valid_t2, "signal_fraction"].sum()),
        "brd_full_decay_relative_residual": manifest["full_positive_time_relative_residual"],
        "source_manifest": str((BRD / "run_manifest.json").resolve()),
    }
    (OUTPUT / "diagnostic_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
