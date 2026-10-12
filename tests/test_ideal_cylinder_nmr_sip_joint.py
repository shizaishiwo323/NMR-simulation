import numpy as np

from advanced_tools.run_ideal_cylinder_nmr_sip_joint import (
    ION_DIFFUSION_M2_S,
    RADIUS_GRID_UM,
    RADII_UM,
    bessel_t2_ms,
    fit_joint_distribution,
    joint_problem,
    second_difference_matrix,
)


def test_ideal_cylinder_analytical_benchmarks():
    t2_ms = np.array([bessel_t2_ms(float(radius))[0] for radius in RADII_UM])
    tau_s = RADII_UM**2 * 1e-12 / (2.0 * ION_DIFFUSION_M2_S)

    assert np.allclose(t2_ms, [200.3, 501.6, 1006.3, 2025.1], rtol=2e-3)
    assert np.allclose(tau_s, [0.0015384615, 0.0096153846, 0.0384615385, 0.1538461538], rtol=1e-7)


def test_second_difference_has_constant_nullspace():
    matrix = second_difference_matrix(9)

    assert matrix.shape == (7, 9)
    assert np.allclose(matrix @ np.ones(9), 0.0)


def test_joint_inversion_recovers_one_exact_radius_bin():
    radius_um = float(RADIUS_GRID_UM[40])
    times_s = np.linspace(0.0, 3.0, 50)
    frequencies_hz = np.geomspace(1e-3, 1e5, 50)
    omega = 2.0 * np.pi * frequencies_hz
    t2_s = radius_um / 10.0
    tau_s = (radius_um * 1e-6) ** 2 / (2.0 * ION_DIFFUSION_M2_S)
    sigma0, sigma_infinity, epsilon_infinity = 0.02, 0.0203, 7e-10
    nmr_data = np.exp(-times_s / t2_s)
    sip_data = sigma_infinity - (sigma_infinity - sigma0) / (1.0 + 1j * omega * tau_s) + 1j * omega * epsilon_infinity

    problem = joint_problem(times_s, nmr_data, frequencies_hz, sip_data, sigma0, sigma_infinity, epsilon_infinity)
    result = fit_joint_distribution(problem, beta=1.0, regularization=1e-6)

    assert result["converged"]
    assert RADIUS_GRID_UM[np.argmax(result["weights"])] == radius_um
