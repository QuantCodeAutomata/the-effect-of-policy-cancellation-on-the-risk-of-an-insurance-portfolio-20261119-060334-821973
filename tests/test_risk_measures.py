"""Unit tests for src/risk_measures.py."""
import numpy as np
import pytest
from scipy.stats import norm

from src.risk_measures import (
    rho_phi,
    empirical_var,
    empirical_tvar,
    empirical_risk_measure,
    wasserstein_distance_1d,
)


def test_rho_phi_var_matches_normal_ppf():
    alpha = 0.99
    assert rho_phi("VAR", alpha) == pytest.approx(norm.ppf(alpha))


def test_rho_phi_tvar_formula():
    alpha = 0.99
    z = norm.ppf(alpha)
    expected = norm.pdf(z) / (1 - alpha)
    assert rho_phi("TVAR", alpha) == pytest.approx(expected)


def test_rho_phi_tvar_greater_than_var():
    # TVaR >= VaR at the same confidence level for the normal law.
    assert rho_phi("TVAR", 0.99) > rho_phi("VAR", 0.99)


def test_rho_phi_unknown_measure_raises():
    with pytest.raises(ValueError):
        rho_phi("XYZ", 0.99)


def test_empirical_var_on_standard_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(200_000)
    var99 = empirical_var(sample, 0.99)
    assert var99 == pytest.approx(norm.ppf(0.99), abs=0.05)


def test_empirical_tvar_at_least_var():
    rng = np.random.default_rng(1)
    sample = rng.standard_normal(50_000)
    v = empirical_var(sample, 0.95)
    t = empirical_tvar(sample, 0.95)
    assert t >= v


def test_empirical_tvar_single_tail_point():
    # Degenerate case: only the maximum value is in the tail.
    sample = np.array([1.0, 2.0, 3.0, 100.0])
    t = empirical_tvar(sample, 0.99)
    assert t >= np.quantile(sample, 0.99)


def test_empirical_risk_measure_dispatch():
    rng = np.random.default_rng(2)
    sample = rng.standard_normal(10_000)
    assert empirical_risk_measure(sample, "VAR", 0.9) == pytest.approx(empirical_var(sample, 0.9))
    assert empirical_risk_measure(sample, "TVAR", 0.9) == pytest.approx(empirical_tvar(sample, 0.9))
    with pytest.raises(ValueError):
        empirical_risk_measure(sample, "nope", 0.9)


def test_wasserstein_distance_zero_for_identical_samples():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(1000)
    assert wasserstein_distance_1d(s, s, p=1) == pytest.approx(0.0, abs=1e-10)


def test_wasserstein_distance_positive_for_shifted_samples():
    rng = np.random.default_rng(4)
    s1 = rng.standard_normal(5000)
    s2 = rng.standard_normal(5000) + 1.0
    w1 = wasserstein_distance_1d(s1, s2, p=1)
    # For a pure location shift the 1D Wasserstein-1 distance equals the shift.
    assert w1 == pytest.approx(1.0, abs=0.05)


def test_wasserstein_distance_handles_unequal_sizes():
    rng = np.random.default_rng(5)
    s1 = rng.standard_normal(1000)
    s2 = rng.standard_normal(3000)
    w = wasserstein_distance_1d(s1, s2, p=2)
    assert w >= 0.0
