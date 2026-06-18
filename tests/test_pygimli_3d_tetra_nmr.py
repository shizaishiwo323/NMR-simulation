from pathlib import Path

import numpy as np

from advanced_tools.three_d_nmr.open_pore_workflow import detect_open_pores
from advanced_tools.three_d_nmr.pygimli_3d_tetra_nmr import (
    NativeMeshingResourcePause,
    TetraNmrParams,
    assemble_tetra_fem_system,
    assert_native_pygimli_mesher_available,
    build_native_pygimli_plc,
    build_voxel_tetra_mesh,
    call_tetgen_poly,
    estimate_fem_resource_usage,
    mesh_summary_frame,
    parse_tetgen_statistics,
    parse_args,
    save_decay_plot,
    save_tetra_mesh_geometry,
    save_tetra_mesh_preview,
    select_simulation_mask,
    solve_tetra_decay,
    write_tetgen_poly_from_arrays,
)
from advanced_tools.three_d_nmr.render_pygimli_tetra_mesh_html import load_render_arrays, write_html


def test_voxel_tetra_mesh_counts_for_open_tunnel():
    volume = np.full((3, 3, 3), 255, dtype=np.uint8)
    volume[:, 1, 1] = 0
    detection = detect_open_pores(volume, pore_value=0, solid_value=255, connectivity=6)

    mesh = build_voxel_tetra_mesh(detection.open_mask, voxel_size_um=1.7)

    assert detection.open_pore_voxels == 3
    assert mesh.tets.shape == (18, 4)
    assert mesh.nodes_xyz_um.shape[1] == 3
    assert mesh.solid_boundary_faces == 12
    assert mesh.external_boundary_faces == 2
    assert mesh.solid_boundary_triangles.shape == (24, 3)
    assert np.sum(mesh.solid_boundary_area_lumped) > 0.0


def test_tetra_fem_decay_runs_and_writes_plots(tmp_path: Path):
    volume = np.full((3, 3, 3), 255, dtype=np.uint8)
    volume[:, 1, 1] = 0
    detection = detect_open_pores(volume, pore_value=0, solid_value=255, connectivity=6)
    mesh = build_voxel_tetra_mesh(detection.open_mask, voxel_size_um=1.7)
    system = assemble_tetra_fem_system(mesh, chunk_size=10)
    params = TetraNmrParams(voxel_size_um=1.7, dt_ms=5.0, t_max_ms=10.0)

    time_ms, signal, history = solve_tetra_decay(system, params, cg_rtol=1e-8, cg_maxiter=100)
    normalized = signal / signal[0]

    assert time_ms.tolist() == [0.0, 5.0, 10.0]
    assert np.all(np.isfinite(normalized))
    assert normalized[0] == 1.0
    assert normalized[-1] < normalized[0]
    assert history["cg_info"].max() == 0

    mesh_png = tmp_path / "mesh.png"
    decay_png = tmp_path / "decay.png"
    save_tetra_mesh_preview(mesh.nodes_xyz_um, mesh.boundary_triangles_all, mesh_png)
    save_decay_plot(time_ms, normalized, decay_png)

    assert mesh_png.exists() and mesh_png.stat().st_size > 0
    assert decay_png.exists() and decay_png.stat().st_size > 0


def test_mesh_summary_tracks_all_vs_open_pore_domain():
    volume = np.full((4, 4, 4), 255, dtype=np.uint8)
    volume[:, 1, 1] = 0
    volume[2, 2, 2] = 0
    detection = detect_open_pores(volume, pore_value=0, solid_value=255, connectivity=6)
    all_mask = volume == 0
    mesh = build_voxel_tetra_mesh(all_mask, voxel_size_um=1.7)
    params = TetraNmrParams(voxel_size_um=1.7)

    summary = mesh_summary_frame(detection, mesh, None, params, {}, pore_domain="all")

    row = summary.iloc[0]
    assert row["pore_domain"] == "all"
    assert row["responsive_pore_voxels"] == 5
    assert row["open_voxels"] == 4
    assert row["closed_pore_voxels"] == 1


def test_saturated_workflow_defaults_to_all_pore_domain(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["pygimli_3d_tetra_nmr.py"],
    )

    args = parse_args()

    assert args.pore_domain == "all"
    assert args.meshing_backend == "pygimli_native"


def test_select_simulation_mask_all_keeps_closed_pores():
    volume = np.full((4, 4, 4), 255, dtype=np.uint8)
    volume[:, 1, 1] = 0
    volume[2, 2, 2] = 0
    detection = detect_open_pores(volume, pore_value=0, solid_value=255, connectivity=6)

    all_mask, responsive_voxels = select_simulation_mask(
        volume,
        detection,
        pore_domain="all",
        pore_value=0,
    )

    assert responsive_voxels == 5
    assert np.count_nonzero(all_mask) == 5
    assert all_mask[2, 2, 2]


def test_native_pygimli_mesher_preflight_rejects_missing_tetgen():
    missing_name = "definitely_missing_tetgen_for_test"

    try:
        assert_native_pygimli_mesher_available(tetgen_executable=missing_name)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing TetGen executable to raise RuntimeError")

    assert "pyGIMLi native 3D meshing requires TetGen" in message
    assert missing_name in message


def test_native_surface_guard_reports_structured_pause_details():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True

    try:
        build_native_pygimli_plc(
            mask,
            voxel_size_um=1.0,
            max_surface_triangles=1,
            max_region_markers=10,
        )
    except NativeMeshingResourcePause as exc:
        pause = exc
    else:
        raise AssertionError("expected native surface guard to raise NativeMeshingResourcePause")

    assert pause.details["pause_stage"] == "surface_triangle_guard"
    assert pause.details["surface_triangles"] > 1
    assert pause.details["max_surface_triangles"] == 1
    assert pause.details["downsampling"] == "none"


def test_direct_tetgen_poly_writer_records_nodes_facets_and_regions(tmp_path: Path):
    nodes_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int32,
    )
    regions = np.array([[0.25, 0.25, 0.25]], dtype=float)
    poly_path = tmp_path / "single_tet.poly"

    stats = write_tetgen_poly_from_arrays(poly_path, nodes_xyz, faces, regions)

    text = poly_path.read_text(encoding="utf-8")
    assert stats["poly_nodes"] == 4
    assert stats["poly_facets"] == 4
    assert stats["poly_region_markers"] == 1
    assert "4\t3\t0\t1" in text
    assert "4\t1" in text
    assert "0\t0.25\t0.25\t0.25\t1\t0" in text


def test_parse_tetgen_statistics_extracts_mesh_size():
    text = """
Statistics:

  Input points: 1862942
  Input facets: 3732044
  Mesh points: 3292926
  Mesh tetrahedra: 13758957
  Mesh faces: 29598709
"""

    stats = parse_tetgen_statistics(text)

    assert stats["input_points"] == 1862942
    assert stats["input_facets"] == 3732044
    assert stats["mesh_points"] == 3292926
    assert stats["mesh_tetrahedra"] == 13758957
    assert stats["mesh_faces"] == 29598709


def test_call_tetgen_poly_runs_from_poly_directory_with_filename(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    relative_dir = Path("nested")
    relative_dir.mkdir()
    poly_path = relative_dir / "mesh.poly"
    poly_path.write_text("", encoding="utf-8")
    recorded = {}

    class Completed:
        returncode = 0
        stdout = "  Mesh tetrahedra: 12\n"

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded["cwd"] = kwargs["cwd"]
        return Completed()

    monkeypatch.setattr(
        "advanced_tools.three_d_nmr.pygimli_3d_tetra_nmr.subprocess.run",
        fake_run,
    )

    base_path, _, stats = call_tetgen_poly(
        poly_path,
        tetgen_executable="tetgen",
        quality=1.2,
        max_cell_volume_um3=0.0,
    )

    assert recorded["command"][-1] == poly_path.name
    assert recorded["cwd"] == str((tmp_path / relative_dir).resolve())
    assert base_path == (tmp_path / relative_dir / "mesh.1").resolve()
    assert stats["mesh_tetrahedra"] == 12


def test_estimate_fem_resource_usage_reports_existing_assembly_lower_bound():
    estimate = estimate_fem_resource_usage(mesh_points=100, mesh_tetrahedra=10)

    assert estimate["mesh_points"] == 100
    assert estimate["mesh_tetrahedra"] == 10
    assert estimate["coo_stiffness_entries"] == 160
    assert estimate["assembly_lower_bound_bytes"] > 0
    assert estimate["assembly_lower_bound_gb"] == estimate["assembly_lower_bound_bytes"] / 1024**3
    assert "COO stiffness staging" in estimate["basis"]


def test_mesh_geometry_html_renderer_writes_file(tmp_path: Path):
    volume = np.full((3, 3, 3), 255, dtype=np.uint8)
    volume[:, 1, 1] = 0
    detection = detect_open_pores(volume, pore_value=0, solid_value=255, connectivity=6)
    mesh = build_voxel_tetra_mesh(detection.open_mask, voxel_size_um=1.7)
    geometry_npz = tmp_path / "mesh_geometry.npz"
    html_path = tmp_path / "mesh.html"

    save_tetra_mesh_geometry(mesh, geometry_npz)
    vertices, normals, colors, metadata = load_render_arrays(
        geometry_npz,
        max_triangles=20,
        include_external=True,
    )
    write_html(html_path, title="test mesh", vertices=vertices, normals=normals, colors=colors, metadata=metadata)

    assert geometry_npz.exists() and geometry_npz.stat().st_size > 0
    assert html_path.exists() and html_path.stat().st_size > 0
    assert metadata["rendered_triangle_count"] <= 20
