"""Edgeworth and Cornish-Fisher expansions for the constant cancellation
regime (Section 3 of the paper).

Hermite polynomials are provided by ``scipy.special.eval_hermitenorm``
(probabilist's / statistician's convention H_k(x) = (-1)^k phi^{(k)}(x)/phi(x),
matching Lemma 3.2's definition exactly), so they are not reimplemented here.
The Edgeworth density correction, Cornish-Fisher expansion and the
Theorem 3.3 / 3.8 asymptotic Delta formulas are custom implementations of the
paper's explicit equations.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_hermitenorm
from scipy.stats import norm


def hermite(k: int, x: np.ndarray) -> np.ndarray:
    """Probabilist's Hermite polynomial H_k(x) (Lemma 3.2)."""
    return eval_hermitenorm(k, x)


def edgeworth_cdf(x: np.ndarray, n: int, kappa3: float, kappa4: float) -> np.ndarray:
    """Classical Edgeworth expansion F_n(x) (Lemma 3.2, eq. 3.1):

    F_n(x) = Phi(x) + phi(x) [ R_{1/2}(x)/sqrt(n) + R_1(x)/n + o(1/n) ],
    R_{1/2}(x) = -(kappa_3/6) H_2(x),
    R_1(x) = -(kappa_4/24) H_3(x) - (kappa_3^2/72) H_5(x).
    """
    x = np.asarray(x, dtype=np.float64)
    r_half = -(kappa3 / 6.0) * hermite(2, x)
    r_one = -(kappa4 / 24.0) * hermite(3, x) - (kappa3 ** 2 / 72.0) * hermite(5, x)
    return norm.cdf(x) + norm.pdf(x) * (r_half / np.sqrt(n) + r_one / n)


def cornish_fisher_quantile(alpha: float, n: int, kappa3: float, kappa4: float) -> float:
    """Cornish-Fisher expansion F_n^{-1}(alpha) (eq. 3.2):

    F_n^{-1}(alpha) = Phi^{-1}(alpha) + Q_{1/2}(u)/sqrt(n) + Q_1(u)/n + o(1/n),
    u = Phi^{-1}(alpha),
    Q_{1/2}(u) = (kappa_3/6)(u^2 - 1),
    Q_1(u) = (kappa_4/24)(u^3 - 3u) - (kappa_3^2/36)(2u^3 - 5u).
    """
    u = norm.ppf(alpha)
    q_half = (kappa3 / 6.0) * (u ** 2 - 1.0)
    q_one = (kappa4 / 24.0) * (u ** 3 - 3.0 * u) - (kappa3 ** 2 / 36.0) * (2.0 * u ** 3 - 5.0 * u)
    return float(u + q_half / np.sqrt(n) + q_one / n)


def delta_mn_var_theory(m: int, n: int, sigma: float, alpha: float) -> float:
    """Theorem 3.3: Delta_{m,n}(X, VaR_alpha) = m*sigma*Phi^{-1}(alpha) / (2 n^{3/2})."""
    z = norm.ppf(alpha)
    return m * sigma * z / (2.0 * n ** 1.5)


def delta_mn_rho_theory(m: int, n: int, sigma: float, rho_phi_value: float) -> float:
    """Theorem 3.8 / 3.12: Delta_{m,n}(X, rho) = m*sigma*rho(Phi) / (2 n^{3/2})."""
    return m * sigma * rho_phi_value / (2.0 * n ** 1.5)


def hermite_probability_measures(x: np.ndarray, kappa3: float, kappa4: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalized Edgeworth signed measures P_0, P_1, P_2 (Definition 3.5 /
    Theorem 3.6), evaluated at points ``x`` (density form, used by the
    generalized expansion underlying the L^1-Lipschitz Theorem 3.8):

    P_0(-phi|kappa)[x] = phi(x),
    P_1(-phi|kappa)[x] = phi(x) (kappa_3/6) H_3(x),
    P_2(-phi|kappa)[x] = phi(x) ((kappa_4/24) H_4(x) + (kappa_3^2/72) H_6(x)),
    H_6(x) = x^6 - 15x^4 + 45x^2 - 15.
    """
    x = np.asarray(x, dtype=np.float64)
    phi = norm.pdf(x)
    p0 = phi
    p1 = phi * (kappa3 / 6.0) * hermite(3, x)
    h6 = x ** 6 - 15.0 * x ** 4 + 45.0 * x ** 2 - 15.0
    p2 = phi * ((kappa4 / 24.0) * hermite(4, x) + (kappa3 ** 2 / 72.0) * h6)
    return p0, p1, p2


def power_law_fit(n_values: np.ndarray, delta_values: np.ndarray) -> tuple[float, float]:
    """Fit Delta_n ~ C * n^{-beta} by ordinary least squares in log-log space.

    Returns (C, beta) such that delta ~ C * n**(-beta). Only points with a
    strictly positive delta are used (exp_1 methodology step 8).
    """
    n_values = np.asarray(n_values, dtype=np.float64)
    delta_values = np.asarray(delta_values, dtype=np.float64)
    mask = delta_values > 0
    log_n = np.log(n_values[mask])
    log_d = np.log(delta_values[mask])
    beta_neg, log_c = np.polyfit(log_n, log_d, 1)
    return float(np.exp(log_c)), float(-beta_neg)
