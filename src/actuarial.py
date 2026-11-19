"""Hattendorff's theorem backward recursion for term life and pure endowment
insurance (Section 5).

Custom implementation of Theorem 5.1 (Hattendorff's theorem for Markov
chains), specialised to the two-state (alive/dead) term life recursion
(eq. 5.5) and pure endowment recursion (eq. 5.6), together with the
standalone and aggregated cancellation-cost approximations of eq. (5.1) and
(5.2), which apply Lemma 4.1 / Theorem 4.3 (standalone) and Lemma 4.5 /
Corollary 4.6 (aggregated) to the resulting segment variances.
"""

from __future__ import annotations

import numpy as np

from src.proportional import f_theta
from src.heterogeneous import effective_cancellation_rate


def net_premium_term_life(q: np.ndarray, v: float, benefit: float) -> float:
    """Net annual premium for a term life contract by the equivalence
    principle (Section 5.2): the expected present value of premium income
    equals the expected present value of the death benefit.

    P = B * sum_k v^{k+1} * (prod_{j<k} p_{x+j}) * q_{x+k}
        / sum_k v^k * (prod_{j<k} p_{x+j}),   k = 0, ..., T-1,
    where the death benefit for a death during policy year k+1 is paid one
    period after the year's start, while the premium is paid at the start.
    """
    q = np.asarray(q, dtype=np.float64)
    p = 1.0 - q
    survival = np.concatenate([[1.0], np.cumprod(p)])[:-1]  # prod_{j<k} p_{x+j}, k=0..T-1
    annuity_discount = v ** np.arange(q.size)  # premium paid at the start of year k+1: v^k
    benefit_discount = v ** np.arange(1, q.size + 1)  # death benefit paid at the end of year k+1: v^{k+1}
    numerator = benefit * np.sum(benefit_discount * survival * q)
    denominator = np.sum(annuity_discount * survival)
    return float(numerator / denominator)


def net_premium_pure_endowment(q: np.ndarray, v: float, benefit: float) -> float:
    """Net annual premium for a pure endowment contract by the equivalence
    principle (Section 5.3): the expected present value of premium income
    equals the discounted survival benefit.

    P = B * v^T * prod_{j=0}^{T-1} p_{x+j} / sum_k v^k * prod_{j<k} p_{x+j}.
    """
    q = np.asarray(q, dtype=np.float64)
    p = 1.0 - q
    t = q.size
    survival_to_t = np.prod(p)
    survival = np.concatenate([[1.0], np.cumprod(p)])[:-1]
    discount = v ** np.arange(t)
    denominator = np.sum(discount * survival)
    return float(benefit * v ** t * survival_to_t / denominator)


def term_life_recursion(q: np.ndarray, v: float, benefit: float, premium: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward recursion (eq. 5.5) for the term life prospective reserve
    V_k(a) and conditional variance Var(L_k | J_k = a), k = 0, ..., T.

    V_T(a) = 0, V_k(a) = -P + v (q_{x+k} B + p_{x+k} V_{k+1}(a)),
    Var(L_T|J_T=a) = 0,
    Var(L_k|J_k=a) = v^2 ( q_{x+k} p_{x+k} (B - V_{k+1}(a))^2
                            + p_{x+k} Var(L_{k+1}|J_{k+1}=a) ).
    """
    q = np.asarray(q, dtype=np.float64)
    p = 1.0 - q
    t = q.size
    V = np.zeros(t + 1)
    Var = np.zeros(t + 1)
    V[t] = 0.0
    Var[t] = 0.0
    for k in range(t - 1, -1, -1):
        V[k] = -premium + v * (q[k] * benefit + p[k] * V[k + 1])
        Var[k] = v ** 2 * (
            q[k] * p[k] * (benefit - V[k + 1]) ** 2 + p[k] * Var[k + 1]
        )
    return V, Var


def pure_endowment_recursion(q: np.ndarray, v: float, benefit: float, premium: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward recursion (eq. 5.6) for the pure endowment prospective
    reserve V_k(a) and conditional variance Var(L_k | J_k = a), k = 0,...,T.

    V_T(a) = B, V_k(a) = -P + v p_{x+k} V_{k+1}(a),
    Var(L_T|J_T=a) = 0,
    Var(L_k|J_k=a) = v^2 ( q_{x+k} p_{x+k} V_{k+1}(a)^2
                            + p_{x+k} Var(L_{k+1}|J_{k+1}=a) ).
    """
    q = np.asarray(q, dtype=np.float64)
    p = 1.0 - q
    t = q.size
    V = np.zeros(t + 1)
    Var = np.zeros(t + 1)
    V[t] = benefit
    Var[t] = 0.0
    for k in range(t - 1, -1, -1):
        V[k] = -premium + v * p[k] * V[k + 1]
        Var[k] = v ** 2 * (
            q[k] * p[k] * V[k + 1] ** 2 + p[k] * Var[k + 1]
        )
    return V, Var


def standalone_ccr(sigma_k: float, n_k: float, theta: float, rho_phi_value: float) -> float:
    """Standalone cancellation cost CCR_hat_k (eq. 5.1), applying Lemma 4.1 /
    Theorem 4.3 to a single new-business cohort:

    CCR_hat_k = ((1-theta)/theta) * (sigma_k / sqrt(n_k)) * f(theta) * rho(Phi).
    """
    return float(((1.0 - theta) / theta) * (sigma_k / np.sqrt(n_k)) * f_theta(theta) * rho_phi_value)


def aggregated_ccr(
    sigma_j2: float,
    theta_j: float,
    weights: np.ndarray,
    sigmas2: np.ndarray,
    thetas: np.ndarray,
    n: float,
    rho_phi_value: float,
) -> float:
    """Aggregated cancellation cost CCR_hat_j^aggr (eq. 5.2), applying
    Lemma 4.5 / Corollary 4.6 to a segment j embedded in a portfolio of
    independent cohorts:

    CCR_hat_j^aggr = ((1-theta_j)/theta_j)
        * (sigma_j^2 / sqrt(n <w, sigma^2>)) * f(theta_bar_sigma) * rho(Phi),
    theta_bar_sigma = sum_i pi_i theta_i, pi_i = w_i sigma_i^2 / <w, sigma^2>.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sigmas2 = np.asarray(sigmas2, dtype=np.float64)
    mean_w_sigma2 = float(np.sum(weights * sigmas2))
    theta_bar = effective_cancellation_rate(weights, np.sqrt(sigmas2), thetas)
    return float(
        ((1.0 - theta_j) / theta_j)
        * (sigma_j2 / np.sqrt(n * mean_w_sigma2))
        * f_theta(theta_bar)
        * rho_phi_value
    )


def build_aggregated_portfolio(
    cohort_size: float, theta: float, n_cohorts: int, variance_by_remaining_maturity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and variances of the ``n_cohorts`` annual cohorts that make up
    the aggregated portfolio at the valuation date (Section 5.2 / 5.3).

    Cohort j (remaining maturity j = 1, ..., n_cohorts-1, excluding the
    maturity-0 cohort which carries no future risk) has been in force for
    (T - j) years and has therefore been reduced by cancellations to size
    n_j = cohort_size * (1 - theta)^{T - j}; its variance is the conditional
    variance of the future loss at the point in the recursion corresponding
    to that remaining maturity, ``variance_by_remaining_maturity[j]``.

    Returns (weights w_j, variances sigma_j^2) for j = 1, ..., n_cohorts - 1,
    normalised so that the weights sum to one.
    """
    maturities = np.arange(1, n_cohorts)  # remaining maturity j = 1..n_cohorts-1
    t = n_cohorts
    sizes = cohort_size * (1.0 - theta) ** (t - maturities)
    weights = sizes / sizes.sum()
    variances = variance_by_remaining_maturity[maturities]
    return weights, variances
