from advanced_tools.batch_tiff_random_slices_nmr import average_signal_name, slice_stem


def test_sample_name_controls_slice_and_average_output_names():
    assert slice_stem("sample16", 7) == "sample16_slice_0007"
    assert average_signal_name("sample16", 10) == "sample16_average_10_slices"
