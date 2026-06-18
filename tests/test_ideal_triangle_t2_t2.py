from pathlib import Path

from advanced_tools.run_ideal_triangle_t2_t2 import (
    IdealTriangleT2T2Config,
    build_coupled_triangle_mesh,
    build_run_manifest,
    console_path,
    summarize_mesh,
)


def test_default_config_records_fixed_inversion_parameters(tmp_path: Path):
    cfg = IdealTriangleT2T2Config(output_dir=tmp_path)

    manifest = build_run_manifest(cfg, outputs={}, mesh_summary={})

    assert manifest["model_dimension"] == "2D"
    assert manifest["geometry"]["description"] == "two coupled equilateral triangular pores"
    assert manifest["inversion"]["mode"] == "fixed"
    assert manifest["inversion"]["t2_alpha"] == 1.0
    assert manifest["inversion"]["t2_t2_alpha"] == 0.05


def test_coupled_triangle_mesh_is_nonempty(tmp_path: Path):
    cfg = IdealTriangleT2T2Config(output_dir=tmp_path)

    mesh = build_coupled_triangle_mesh(cfg)
    summary = summarize_mesh(mesh)

    assert summary["node_count"] > 0
    assert summary["cell_count"] > 0
    assert summary["boundary_count"] > 0


def test_console_path_is_encodable_for_windows_cp1252_logs():
    cwd = Path("C:/Users/imgw/Documents/Codex/NMR模拟")
    raw_path = cwd / "simulation_outputs" / "ideal_triangle_t2_t2" / "run_manifest.json"

    display_path = console_path(raw_path, cwd=cwd, encoding="cp1252")

    display_path.encode("cp1252")
    assert display_path == str(Path("simulation_outputs") / "ideal_triangle_t2_t2" / "run_manifest.json")
