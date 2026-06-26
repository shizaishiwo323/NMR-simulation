import numpy as np
import pandas as pd

from advanced_tools.sample16_select_t2_matching_slices import (
    build_spectrum_loss_weights,
    combine_weighted_spectra,
    load_candidate_decays,
    normalize_nonnegative_weights,
    spectrum_fit_metrics,
)


def test_normalize_nonnegative_weights_drops_tiny_and_sums_to_one():
    weights = normalize_nonnegative_weights(np.array([0.4, -0.1, 0.00001, 0.6]), threshold=0.001)

    assert np.allclose(weights, [0.4, 0.0, 0.0, 0.6])
    assert np.isclose(weights.sum(), 1.0)


def test_spectrum_fit_metrics_reports_rmse_and_correlation():
    target = np.array([0.0, 0.5, 1.0])
    candidate = np.array([0.0, 0.25, 1.0])

    metrics = spectrum_fit_metrics(candidate, target)

    assert np.isclose(metrics["rmse"], np.sqrt(((0.0**2) + (0.25**2) + (0.0**2)) / 3.0))
    assert 0.9 < metrics["correlation"] <= 1.0


def test_build_spectrum_loss_weights_emphasizes_secondary_peak_ranges():
    t2_ms = np.array([1.0, 45.0, 270.0, 1500.0])

    weights = build_spectrum_loss_weights(t2_ms)

    assert weights[1] > weights[3]
    assert weights[2] > weights[3]


def test_load_candidate_decays_falls_back_to_slice_decay_files(tmp_path):
    pd.DataFrame({"time_ms": [1.0, 2.0], "normalized_signal": [1.0, 0.8]}).to_csv(
        tmp_path / "sample_slice_0012_nmr_decay.csv",
        index=False,
    )

    time_ms, candidates = load_candidate_decays(tmp_path)

    assert np.allclose(time_ms, [1.0, 2.0])
    assert candidates[0][1] == 12


def test_combine_weighted_spectra_normalizes_mixture_peak():
    spectra = np.array([[0.0, 0.2], [1.0, 0.4], [0.0, 0.8]])
    weights = np.array([0.25, 0.75])

    combined = combine_weighted_spectra(spectra, weights)

    assert np.isclose(combined.max(), 1.0)
    assert combined[2] > combined[0]
