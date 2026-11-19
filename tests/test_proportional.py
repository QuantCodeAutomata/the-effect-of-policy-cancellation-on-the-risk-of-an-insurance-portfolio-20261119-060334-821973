"""Unit tests for src/proportional.py (Section 4.1-4.2, Section 2.2 Example 2.5)."""
import numpy as np
import pytest

from src.proportional import (
    lambda_theta,
    f_theta,
    delta_theta_n_theory,
    ccr_theta_n_theory,
    ccr_small_theta_approximation,
    lambda_theta_max_location,
)


def test_lambda_theta_vanishes_at_endpoints():
    assert lambda_theta(0.0) == pytest.approx(0.0, abs=1e-10)
    assert lambda_theta(1.0) == pytest.approx(0.0, abs=1e-10)


def test_lambda_theta_positive_in_interior():
    thetas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.all(lambda_theta(thetas) > 0)


def test_lambda_theta_maximized_at_three_quarters():
    theta_star, lam_star = lambda_theta_max_location()
    assert theta_star == pytest.approx(0.75)
    assert lam_star == pytest.approx(0.25, abs=1e-10)
    grid = np.linspace(0.001, 0.999, 2001)
    assert np.max(lambda_theta(grid)) <= lam_star + 1e-6
    assert np.argmax(lambda_theta(grid)) == np.argmin(np.abs(grid - 0.75))


def test_f_theta_positive_and_increasing():
    thetas = np.array([0.01, 0.1, 0.3, 0.5, 0.9])
    vals = f_theta(thetas)
    assert np.all(vals > 0)
    assert np.all(np.diff(vals) > 0)


def test_delta_theta_n_theory_scales_as_n_power_minus_half():
    theta, sigma, rho = 0.1, 1.0, 2.3263
    d1 = delta_theta_n_theory(theta, 1_000, sigma, rho)
    d2 = delta_theta_n_theory(theta, 4_000, sigma, rho)
    assert d1 / d2 == pytest.approx(np.sqrt(4), rel=1e-6)


def test_ccr_small_theta_matches_f_theta_over_theta_as_theta_to_zero():
    # CCR_{theta,n} = (lambda_theta/theta) rho(Phi) n^{-1/2}; as theta -> 0,
    # lambda_theta/theta -> 1/2 (first-order Taylor coefficient of eq. 1.1).
    n, rho = 1_000_000, 2.3263
    ccr = ccr_theta_n_theory(1e-6, n, rho)
    approx = ccr_small_theta_approximation(1e-6, n, sigma=1.0, rho_phi_value=rho)
    assert ccr == pytest.approx(approx, rel=1e-3)


def test_ccr_small_theta_approximation_matches_exact_for_small_theta():
    n, sigma, rho = 500_000, 1.0, 2.3263
    for theta in (0.01, 0.05):
        exact = ccr_theta_n_theory(theta, n, rho)
        approx = ccr_small_theta_approximation(theta, n, sigma, rho)
        assert exact == pytest.approx(approx, rel=0.02)
