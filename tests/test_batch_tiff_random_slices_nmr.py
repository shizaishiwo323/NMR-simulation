from advanced_tools.batch_tiff_random_slices_nmr import (
    average_signal_name,
    model_dimension_note,
    parse_slice_indices,
    slice_stem,
)


def test_sample_name_controls_slice_and_average_output_names():
    assert slice_stem("sample16", 7) == "sample16_slice_0007"
    assert average_signal_name("sample16", 10) == "sample16_average_10_slices"


def test_parse_slice_indices_accepts_comma_separated_values():
    assert parse_slice_indices("400") == [400]
    assert parse_slice_indices("1, 2,7") == [1, 2, 7]
    assert parse_slice_indices(None) is None


def test_model_dimension_note_uses_actual_slice_count():
    assert "20-slice 2D ensemble" in model_dimension_note(20)
