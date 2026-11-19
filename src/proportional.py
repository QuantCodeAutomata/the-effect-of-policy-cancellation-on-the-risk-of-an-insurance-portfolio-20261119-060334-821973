"""Proportional cancellation regime in a homogeneous portfolio (Section 4.1-4.2).

Custom implementation of Lemma 4.1 / Theorem 4.3 (paper equations in
Section 4.2) and of the Gaussian benchmark relations of Example 2.5
(Section 2.2), which give the leading-order coefficient lambda_theta and the
per-policyholder cancellation cost CCR_{theta,n}.
"""

from __future__ import annotations

import numpy as np


def lambda_theta(theta: np.ndarray | float) -> np.ndarray | float:
    """lambda_theta = sqrt(1 - theta) - (1 - theta) (Section 2.2, Example 2.5)."""
    theta = np.asarray(theta, dtype=np.float64)
    return np.sqrt(1.0 - theta) - (1.0 - theta)


def f_theta(theta: np.ndarray | float) -> np.ndarray | float:
    """f(theta) = (1 - sqrt(1-theta)) / sqrt(1-theta), the coefficient that
    appears in Delta_{theta,n} (Lemma 4.1 / Theorem 4.3)."""
    theta = np.asarray(theta, dtype=np.float64)
    s = np.sqrt(1.0 - theta)
    return (1.0 - s) / s


def delta_theta_n_theory(theta: float, n: int, sigma: float, rho_phi_value: float) -> float:
    """Lemma 4.1 / Theorem 4.3:

    Delta_{theta,n}(X, rho) = f(theta) * sigma * rho(Phi) / sqrt(n) + o(n^{-1/2}),
    f(theta) = (1 - sqrt(1-theta)) / sqrt(1-theta).
    """
    return float(f_theta(theta) * sigma * rho_phi_value / np.sqrt(n))


def ccr_theta_n_theory(theta: float, n: int, rho_phi_value: float) -> float:
    """Per-cancelling-policyholder cancellation cost (Section 2.2, between
    eq. 2.5 and 2.6):

    CCR_{theta,n}(Phi, rho) = (lambda_theta/theta) * rho(Phi) * n^{-1/2} + O(n^{-3/2}).
    """
    return float((lambda_theta(theta) / theta) * rho_phi_value / np.sqrt(n))


def ccr_small_theta_approximation(theta: float, n: int, sigma: float, rho_phi_value: float) -> float:
    """First-order Taylor approximation of CCR for small theta (eq. 1.1):

    CCR_{theta,n} ~= (1/2 - theta/8) * sigma * rho(Phi) * n^{-1/2}.
    """
    return float((0.5 - theta / 8.0) * sigma * rho_phi_value / np.sqrt(n))


def lambda_theta_max_location() -> tuple[float, float]:
    """The coefficient lambda_theta is maximized at theta=3/4, where
    lambda_{3/4} = 1/4 (Section 2.2)."""
    theta_star = 0.75
    return theta_star, float(lambda_theta(theta_star))
