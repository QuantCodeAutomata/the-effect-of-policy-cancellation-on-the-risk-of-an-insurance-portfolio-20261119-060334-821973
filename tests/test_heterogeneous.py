"""Unit tests for src/heterogeneous.py (Section 4.3-4.4, Example 4.7)."""
import numpy as np
import pytest

from src.heterogeneous import (
    segment_risk_contributions,
    effective_cancellation_rate,
    delta_j_aggr_theory,
    delta_j_standalone_theory,
    example_4_7_ratio,
    equal_rate_limit_ratio,
    bernoulli_cancellation_sizes,
)


def test_segment_risk_contributions_sum_to_one():
    w = np.array([0.3, 0.3, 0.4])
    s = np.array([1.0, 2.0, 0.5])
    pi = segment_risk_contributions(w, s)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= 0)


def test_effective_cancellation_rate_equals_common_rate_when_thetas_equal():
    w = np.array([0.2, 0.3, 0.5])
    s = np.array([1.0, 1.5, 0.8])
    theta_bar = effective_cancellation_rate(w, s, thetas=np.array([0.05, 0.05, 0.05]))
    assert theta_bar == pytest.approx(0.05)


def test_effective_cancellation_rate_is_weighted_average_bounds():
    w = np.array([0.5, 0.5])
    s = np.array([1.0, 1.0])
    thetas = np.array([0.01, 0.10])
    theta_bar = effective_cancellation_rate(w, s, thetas)
    assert min(thetas) <= theta_bar <= max(thetas)
    assert theta_bar == pytest.approx(0.055)


def test_example_4_7_ratio_matches_paper_value():
    # Example 4.7 (Section 4.3): w_1=w_2=0.5, sigma_1=sigma_2, theta_1=1%, theta_2=10%.
    w = np.array([0.5, 0.5])
    s = np.array([1.0, 1.0])
    thetas = np.array([0.01, 0.10])
    ratio = example_4_7_ratio(w, s, thetas)
    assert ratio == pytest.approx(4.03, abs=0.01)


def test_equal_rate_limit_ratio_is_sqrt_pi_and_at_most_one():
    w = np.array([0.5, 0.5])
    s = np.array([1.0, 1.0])
    for j in (0, 1):
        r = equal_rate_limit_ratio(j, w, s)
        assert r <= 1.0 + 1e-10
        assert r == pytest.approx(np.sqrt(0.5))


def test_aggregation_helps_when_theta_j_at_least_theta_bar():
    # When segment j's own cancellation rate exceeds the portfolio average,
    # aggregating it should not increase its cancellation cost relative to
    # standalone (research_objectives: "aggregation helps when theta_j >= theta_bar").
    w = np.array([0.5, 0.5])
    s = np.array([1.0, 1.0])
    thetas = np.array([0.01, 0.10])
    n = 100_000
    rho = 2.3263
    delta_2_standalone = delta_j_standalone_theory(1, w, s, thetas, n, rho)
    delta_2_aggr = delta_j_aggr_theory(1, w, s, thetas, n, rho)
    assert delta_2_aggr <= delta_2_standalone


def test_aggregation_hurts_when_theta_j_below_theta_bar():
    w = np.array([0.5, 0.5])
    s = np.array([1.0, 1.0])
    thetas = np.array([0.01, 0.10])
    n = 100_000
    rho = 2.3263
    delta_1_standalone = delta_j_standalone_theory(0, w, s, thetas, n, rho)
    delta_1_aggr = delta_j_aggr_theory(0, w, s, thetas, n, rho)
    assert delta_1_aggr > delta_1_standalone


def test_bernoulli_cancellation_sizes_concentrate_near_deterministic_value():
    rng = np.random.default_rng(42)
    segment_sizes = np.array([100_000, 50_000])
    thetas = np.array([0.1, 0.2])
    remaining = bernoulli_cancellation_sizes(segment_sizes, thetas, n_mc=2000, rng=rng)
    assert remaining.shape == (2000, 2)
    expected = segment_sizes * (1 - thetas)
    mean_remaining = remaining.mean(axis=0)
    assert np.allclose(mean_remaining, expected, rtol=0.01)
