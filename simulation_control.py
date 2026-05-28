"""Central control file for NMR pore-scale simulations.

Edit this file before running the NMR scripts. Units used by the current
simulation scripts are generally micrometers (um) for length and milliseconds
(ms) for time.
"""

SETTINGS = {
    "run": {
        "interactive": False,
        "output_dir": "simulation_outputs/Auto_NMR_controlled",
        "show_figures": False,
    },
    "modules": {
        # Coupled/uncoupled switches.
        # Set one of the first two to False if only one physical mode is needed.
        "uncoupled_t2": True,
        "coupled_t2": True,
        # T2-T2 and D-T2 are more expensive than 1D T2.
        "t2_t2": True,
        "dt2": True,
        "effective_exchange_rate": False,
    },
    "inversion": {
        # "fixed": manually chosen alpha.
        # "l_curve": automatic L-curve regularization.
        "mode": "fixed",
        "alpha": 1.0,
        # Segmented image / glass-bead data should normally use L-curve.
        "image_mode": "l_curve",
    },
    "saturation": {
        "target_saturations": [0.0226, 0.12, 0.153, 0.380, 1.0],
    },
    "geometry": {
        "mode": "verified_triangle",
        "triangle_angles_deg": [60.0, 60.0, 60.0],
        "contact_angle_deg": 0.0,
        "process": "Drainage",
        "slice_path": None,
    },
    "triangle_ca": {
        "enabled": True,
        "output_dir": "simulation_outputs/triangle_60_60_60_contact_0",
        "contact_angle_deg": 0.0,
        "triangle_angles_deg": [60.0, 60.0, 60.0],
        "processes": ["Drainage", "Imbibition"],
        "selected_rm_mode": "default",
    },
    "triangle_full": {
        "output_dir": "simulation_outputs/triangle_full_CA0p0deg_angles_60-60-60",
        # Full PDE mesh is preserved. These only control D-T2 acquisition-grid density.
        "strict_dt2_b_log_points": 9,
        "strict_dt2_te_points": 16,
    },
    "image_slice": {
        "enabled": True,
        # Put the segmented image in data/input and update this path.
        "input_path": "data/input/Result.tif",
        "output_dir": "simulation_outputs/image_slice_Result",
        "pixel_size_um": 1.0,
        # Default labels used by the original glass-bead slice.
        "solid_value": 255,
        "water_value": 102,
        "air_value": 153,
        "outside_value": 0,
        "unknown_values": [204],
        "smooth_radius_px": 2,
        "interface_sigma_px": 1.25,
        # None means use original resolution for PDE.
        "max_grid_size": None,
        "run_t2": True,
        "run_t2_t2": True,
        "run_dt2": True,
    },
    "t2_t2": {
        "t1_points": 25,
        "t2_points": 35,
        "bin_points": 35,
        "t_axis_min_ms": 1.0,
        "t_axis_max_ms": 10 ** 3.1,
        "bin_min_ms": 10 ** 0.5,
        "bin_max_ms": 10 ** 3.5,
        "mixing_time_ms": 30.0,
        "alpha": 0.05,
    },
    "dt2": {
        "b_axis": [0.0],
        "b_log_min": -3.0,
        "b_log_max": 0.55,
        "b_log_points": 13,
        "te_min_ms": 25.0,
        "te_max_ms": 1300.0,
        "te_points": 24,
        "d_log_min": -3.0,
        "d_bin_scale_to_bulk_D": 1.5,
        "d_bins": 32,
        "t2_bin_min_ms": 10 ** 0.5,
        "t2_bin_max_ms": 10 ** 3.5,
        "t2_bins": 40,
        "delta_ms": 20.0,
        "pde_dt_ms": 8.0,
        "grad_axis": "y",
        "alpha": 0.025,
    },
}
