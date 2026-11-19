"""Unit tests for src/actuarial.py (Section 5 Hattendorff recursion)."""
import numpy as np
import pytest

from src import config
from src.actuarial import (
    net_premium_term_life,
    net_premium_pure_endowment,
    term_life_recursion,
    pure_endowment_recursion,
    standalone_ccr,
    aggregated_ccr,
    build_aggregated_portfolio,
)


@pytest.fixture
def q():
    return np.array(config.MORTALITY_Q)


def test_term_life_premium_matches_paper_value(q):
    p = net_premium_term_life(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT)
    assert p == pytest.approx(config.TERM_LIFE_PREMIUM_PAPER, rel=0.01)


def test_pure_endowment_premium_matches_paper_value(q):
    p = net_premium_pure_endowment(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT)
    assert p == pytest.approx(config.PURE_ENDOWMENT_PREMIUM_PAPER, rel=0.01)


def test_term_life_reserve_boundary_conditions(q):
    P = net_premium_term_life(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT)
    V, Var = term_life_recursion(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT, P)
    assert V[-1] == pytest.approx(0.0)
    assert Var[-1] == pytest.approx(0.0)
    # V_0(a) should be (near) zero under the equivalence-principle premium.
    assert V[0] == pytest.approx(0.0, abs=1.0)


def test_term_life_variance_nonnegative_and_matches_paper_magnitude(q):
    P = net_premium_term_life(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT)
    V, Var = term_life_recursion(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT, P)
    assert np.all(Var >= 0)
    # Paper Table 1: Var(L_0 | J_0=a) ~= 288,421,935 (sigma_0 ~= 16,983).
    assert Var[0] == pytest.approx(288_421_935, rel=0.01)


def test_pure_endowment_reserve_boundary_conditions(q):
    P = net_premium_pure_endowment(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT)
    V, Var = pure_endowment_recursion(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT, P)
    assert V[-1] == pytest.approx(config.ENDOWMENT_BENEFIT)
    assert Var[-1] == pytest.approx(0.0)
    assert V[0] == pytest.approx(0.0, abs=1.0)


def test_pure_endowment_variance_matches_paper_magnitude(q):
    P = net_premium_pure_endowment(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT)
    V, Var = pure_endowment_recursion(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT, P)
    assert Var[0] == pytest.approx(122_423_296, rel=0.02)


def test_standalone_ccr_positive_and_decreasing_in_n():
    sigma_k = 17_000.0
    theta = 0.05
    rho = config.VAR_Z
    ccr_small_n = standalone_ccr(sigma_k, n_k=1000, theta=theta, rho_phi_value=rho)
    ccr_large_n = standalone_ccr(sigma_k, n_k=100_000, theta=theta, rho_phi_value=rho)
    assert ccr_small_n > 0
    assert ccr_large_n < ccr_small_n


def test_build_aggregated_portfolio_weights_sum_to_one():
    variances = np.linspace(1e8, 2e8, config.N_AGGREGATED_COHORTS)
    w, sigma2 = build_aggregated_portfolio(
        config.COHORT_SIZE, 0.05, config.N_AGGREGATED_COHORTS, variances
    )
    assert w.sum() == pytest.approx(1.0)
    assert w.size == config.N_AGGREGATED_COHORTS - 1
    assert np.all(w > 0)


def test_aggregated_ccr_smaller_than_standalone_for_equal_theta():
    # Equal cancellation rate across cohorts: aggregation should reduce the
    # relative cancellation cost compared with the standalone cohort cost.
    n_cohorts = config.N_AGGREGATED_COHORTS
    theta = 0.05
    variances = np.linspace(2.9e8, 1.2e8, n_cohorts)  # decreasing with maturity, like Table 1
    w, sigma2 = build_aggregated_portfolio(config.COHORT_SIZE, theta, n_cohorts, variances)
    j = 5
    standalone = standalone_ccr(np.sqrt(sigma2[j]), config.COHORT_SIZE, theta, config.VAR_Z)
    aggr = aggregated_ccr(
        sigma2[j], theta, w, sigma2, np.full_like(w, theta), config.COHORT_SIZE * n_cohorts, config.VAR_Z
    )
    assert aggr < standalone
