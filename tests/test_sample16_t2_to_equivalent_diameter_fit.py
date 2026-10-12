import numpy as np

from advanced_tools.sample16_t2_to_equivalent_diameter_fit import fit_rho2, t2_to_diameter_um
from advanced_tools.sample16_t2_to_characteristic_length_fit import (
    fit_shared_rho2,
    open_cylinder_diameter_to_lc_um,
    sphere_diameter_to_lc_um,
    t2_to_characteristic_length_um,
)


def test_spherical_conversion_and_peak_constrained_fit():
    assert np.isclose(t2_to_diameter_um(1000.0, 3.16), 18.96)

    t2 = np.array([100.0, 1000.0])
    amplitude = np.array([0.1, 1.0])
    pore_diameter = t2_to_diameter_um(t2, 5.0)
    pore_volume = amplitude.copy()
    scan = fit_rho2(t2, amplitude, pore_diameter, pore_volume, 30.0, np.array([3.0, 5.0, 7.0]))

    assert scan.loc[scan["combined_score"].idxmin(), "rho2_um_per_s"] == 5.0


def test_characteristic_length_mapping_and_shared_fit():
    assert np.isclose(t2_to_characteristic_length_um(1000.0, 3.16), 3.16)
    assert np.isclose(sphere_diameter_to_lc_um(6.0), 1.0)
    assert np.isclose(open_cylinder_diameter_to_lc_um(4.0), 1.0)

    t2 = np.array([100.0, 1000.0])
    amplitude = np.array([0.1, 1.0])
    target = t2_to_characteristic_length_um(t2, 5.0)
    scan = fit_shared_rho2(
        t2,
        amplitude,
        target,
        amplitude,
        5.0,
        target,
        amplitude,
        5.0,
        np.array([3.0, 5.0, 7.0]),
    )

    assert scan.loc[scan["combined_volume_weighted_score"].idxmin(), "rho2_um_per_s"] == 5.0
