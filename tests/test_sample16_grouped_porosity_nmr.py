import numpy as np

from advanced_tools.sample16_grouped_porosity_nmr import (
    candidate_indices_for_group,
    group_porosity_series,
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
