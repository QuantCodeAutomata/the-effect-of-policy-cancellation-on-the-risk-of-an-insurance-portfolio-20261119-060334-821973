"""Heterogeneous portfolio and stochastic cancellation (Section 4.3-4.4).

Custom implementation of Lemma 4.5, Example 4.7, Corollary 4.6 and the
Bernoulli cancellation extension of Lemma 4.8.
"""

from __future__ import annotations

import numpy as np

from src.proportional import f_theta


def segment_risk_contributions(weights: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Variance-based risk contribution pi_j = w_j sigma_j^2 / <w, sigma^2>
    (Lemma 4.5)."""
    weights = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    w_sigma2 = weights * sigmas ** 2
    return w_sigma2 / w_sigma2.sum()


def effective_cancellation_rate(weights: np.ndarray, sigmas: np.ndarray, thetas: np.ndarray) -> float:
    """theta_bar_sigma = sum_j pi_j theta_j, the variance-weighted average
    cancellation rate (Lemma 4.5)."""
    pi = segment_risk_contributions(weights, sigmas)
    thetas = np.asarray(thetas, dtype=np.float64)
    return float(np.sum(pi * thetas))


def delta_j_aggr_theory(
    j: int, weights: np.ndarray, sigmas: np.ndarray, thetas: np.ndarray, n: int, rho_phi_value: float
) -> float:
    """Lemma 4.5, eq. (4.2):

    Delta_j^aggr = sigma_j^2 / sqrt(n <w, sigma^2>) * f(theta_bar_sigma) * rho(Phi) + o(n^{-1/2}),
    theta_bar_sigma = sum_i pi_i theta_i, pi_i = w_i sigma_i^2 / <w, sigma^2>.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    w_sigma2 = weights * sigmas ** 2
    mean_w_sigma2 = w_sigma2.sum()
    theta_bar = effective_cancellation_rate(weights, sigmas, thetas)
    return float(
        sigmas[j] ** 2 / np.sqrt(n * mean_w_sigma2) * f_theta(theta_bar) * rho_phi_value
    )


def delta_j_standalone_theory(j: int, weights: np.ndarray, sigmas: np.ndarray, thetas: np.ndarray, n: int, rho_phi_value: float) -> float:
    """Standalone (non-aggregated) segment cancellation effect:

    Delta_j = sigma_j / sqrt(w_j n) * f(theta_j) * rho(Phi),
    i.e. Lemma 4.1 / Theorem 4.3 applied to a segment in isolation with its
    own portfolio size w_j * n (Example 4.7).
    """
    weights = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    n_j = weights[j] * n
    return float(sigmas[j] / np.sqrt(n_j) * f_theta(thetas[j]) * rho_phi_value)


def example_4_7_ratio(weights: np.ndarray, sigmas: np.ndarray, thetas: np.ndarray) -> float:
    """Asymptotic ratio Delta_1^aggr / Delta_1 for Example 4.7 (Section 4.3),
    obtained by taking the ratio of the n-independent coefficients of
    :func:`delta_j_aggr_theory` and :func:`delta_j_standalone_theory`:

    ratio = sqrt(w_1) * f(theta_bar_sigma) / f(theta_1),

    since sigma_1 = sigma_2 cancels and sqrt(n) factors cancel up to the
    sqrt(w_1) weighting of the standalone portfolio size n_1 = w_1 n.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    theta_bar = effective_cancellation_rate(weights, sigmas, thetas)
    return float(np.sqrt(weights[0]) * f_theta(theta_bar) / f_theta(thetas[0]))


def equal_rate_limit_ratio(j: int, weights: np.ndarray, sigmas: np.ndarray) -> float:
    """When all segments share the same cancellation rate theta_j = theta,
    Delta_j^aggr / Delta_j -> sqrt(pi_j) <= 1 (research_objectives)."""
    pi = segment_risk_contributions(weights, sigmas)
    return float(np.sqrt(pi[j]))


def bernoulli_cancellation_sizes(
    segment_sizes: np.ndarray, thetas: np.ndarray, n_mc: int, rng: np.random.Generator
) -> np.ndarray:
    """Stochastic (Bernoulli) cancellation (Lemma 4.8): for each segment j and
    each Monte Carlo replication, draw n_j independent Bernoulli(theta_j)
    cancellation indicators and return the number of *remaining* policies
    ``Ntilde_j`` for each replication (shape (n_mc, l)).

    By the strong law of large numbers, Ntilde_j / n_j -> 1 - theta_j a.s.,
    so the stochastic remaining size concentrates around the deterministic
    proportional-cancellation size (1-theta_j) n_j.
    """
    segment_sizes = np.asarray(segment_sizes, dtype=np.int64)
    thetas = np.asarray(thetas, dtype=np.float64)
    l = segment_sizes.size
    remaining = np.zeros((n_mc, l), dtype=np.int64)
    for j in range(l):
        cancelled = rng.binomial(segment_sizes[j], thetas[j], size=n_mc)
        remaining[:, j] = segment_sizes[j] - cancelled
    return remaining
