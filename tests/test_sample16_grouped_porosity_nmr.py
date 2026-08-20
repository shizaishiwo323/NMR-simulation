import numpy as np

from advanced_tools.sample16_grouped_porosity_nmr import (
    candidate_indices_for_group,
    completed_group_run,
    group_porosity_series,
    load_pore_size_table,
)


def test_group_porosity_series_limits_group_range():
    porosity = [0.10, 0.105, 0.109, 0.122, 0.125]

    groups = group_porosity_series(porosity, threshold=0.01)

    assert [(g.start_index, g.end_index, g.representative_index, g.slice_count) for g in groups] == [
        (0, 2, 1, 3),
        (3, 4, 3, 2),
    ]
    assert np.isclose(groups[0].porosity_max - groups[0].porosity_min, 0.009)


def test_candidate_indices_for_group_orders_middle_then_neighbors():
    assert candidate_indices_for_group(10, 14) == [12, 11, 13, 10, 14]
    assert candidate_indices_for_group(10, 13) == [11, 12, 10, 13]


def test_load_pore_size_table_accepts_pnextract_parsed_csv(tmp_path):
    pores_csv = tmp_path / "pores.csv"
    pores_csv.write_text(
        "pore_id,pore_volume_m3,pore_radius_m\n"
        "1,2e-18,3e-6\n",
        encoding="utf-8",
    )

    pores = load_pore_size_table(pores_csv)

    assert np.isclose(pores.loc[0, "pore_diameter_um"], 6.0)
    assert np.isclose(pores.loc[0, "pore_volume_um3"], 2.0)


def test_completed_group_run_requires_decay_manifest_and_mesh(tmp_path):
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "average_normalized_decay.csv").write_text("time_ms,average_normalized_signal\n0,1\n", encoding="utf-8")
    (tmp_path / "slice_triangular_mesh.png").write_bytes(b"png")

    assert completed_group_run(tmp_path)
