import numpy as np

from advanced_tools.sample16_rho2_cumulative_vc_calibration import calibration


def test_vc1_alignment_uses_distribution_endpoints() -> None:
    result = calibration(
        pore_radius_um=np.array([1.0, 4.0]),
        pore_volume=np.array([1.0, 1.0]),
        t2_ms=np.array([10.0, 20.0]),
        amplitude=np.array([1.0, 1.0]),
        ct_porosity_pct=20.0,
        total_porosity_pct=25.0,
        alignment_vc=1.0,
    )

    assert result["raw_ct_quantile_at_alignment"] == 1.0
    assert result["ct_radius_at_alignment_um"] == 4.0
    assert np.isclose(result["nmr_t2_at_alignment_ms"], 20.0)
    assert np.isclose(result["rho2_sphere_alpha3_um_per_s"], 4.0 / 0.06)
