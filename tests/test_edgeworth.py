"""Unit tests for src/edgeworth.py (Section 3 Edgeworth / Cornish-Fisher)."""
import numpy as np
import pytest
from scipy.stats import norm

from src.edgeworth import (
    hermite,
    edgeworth_cdf,
    cornish_fisher_quantile,
    delta_mn_var_theory,
    delta_mn_rho_theory,
    hermite_probability_measures,
    power_law_fit,
)


def test_hermite_small_orders_match_closed_form():
    x = np.array([-1.5, 0.0, 0.7, 2.3])
    assert np.allclose(hermite(0, x), np.ones_like(x))
    assert np.allclose(hermite(1, x), x)
    assert np.allclose(hermite(2, x), x ** 2 - 1)
    assert np.allclose(hermite(3, x), x ** 3 - 3 * x)


def test_edgeworth_reduces_to_gaussian_when_cumulants_zero():
    x = np.linspace(-3, 3, 50)
    f = edgeworth_cdf(x, n=100, kappa3=0.0, kappa4=0.0)
    assert np.allclose(f, norm.cdf(x))


def test_cornish_fisher_reduces_to_gaussian_quantile_when_cumulants_zero():
    q = cornish_fisher_quantile(0.99, n=500, kappa3=0.0, kappa4=0.0)
    assert q == pytest.approx(norm.ppf(0.99))


def test_cornish_fisher_correction_vanishes_as_n_grows():
    # The non-Gaussian correction should shrink as n increases (asymptotic expansion).
    q_small = cornish_fisher_quantile(0.99, n=100, kappa3=1.0, kappa4=2.0)
    q_large = cornish_fisher_quantile(0.99, n=100_000, kappa3=1.0, kappa4=2.0)
    z = norm.ppf(0.99)
    assert abs(q_large - z) < abs(q_small - z)


def test_delta_mn_var_theory_scales_as_n_power_minus_1p5():
    m, sigma, alpha = 5, 1.0, 0.99
    d1 = delta_mn_var_theory(m, 1000, sigma, alpha)
    d2 = delta_mn_var_theory(m, 8000, sigma, alpha)  # n multiplied by 8 -> delta / 8^1.5 = /22.6
    ratio = d1 / d2
    assert ratio == pytest.approx(8 ** 1.5, rel=1e-6)


def test_delta_mn_var_theory_linear_in_m():
    sigma, alpha, n = 1.0, 0.99, 10_000
    d1 = delta_mn_var_theory(1, n, sigma, alpha)
    d5 = delta_mn_var_theory(5, n, sigma, alpha)
    assert d5 == pytest.approx(5 * d1)


def test_delta_mn_rho_theory_matches_var_case():
    m, n, sigma, alpha = 3, 20_000, 1.0, 0.99
    z = norm.ppf(alpha)
    assert delta_mn_rho_theory(m, n, sigma, z) == pytest.approx(delta_mn_var_theory(m, n, sigma, alpha))


def test_hermite_probability_measures_shapes_and_p0_is_density():
    x = np.linspace(-2, 2, 10)
    p0, p1, p2 = hermite_probability_measures(x, kappa3=0.3, kappa4=0.5)
    assert p0.shape == x.shape
    assert np.allclose(p0, norm.pdf(x))


def test_power_law_fit_recovers_known_exponent():
    n = np.array([1e3, 1e4, 1e5, 1e6])
    beta_true, c_true = 1.5, 2.0
    delta = c_true * n ** (-beta_true)
    c_hat, beta_hat = power_law_fit(n, delta)
    assert beta_hat == pytest.approx(beta_true, rel=1e-6)
    assert c_hat == pytest.approx(c_true, rel=1e-6)
