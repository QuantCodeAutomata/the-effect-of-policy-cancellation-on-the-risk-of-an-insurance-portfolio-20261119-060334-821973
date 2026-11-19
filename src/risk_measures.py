"""Empirical and Gaussian-benchmark risk measures used throughout the paper.

Implements the quantities rho(Phi) (the value of a risk measure at the
standard normal distribution, Section 2.2 Example 2.5 and Section 3.3) and
empirical VaR / TVaR estimators (implementation_notes of exp_1 / exp_2), plus
a one-dimensional Wasserstein distance via the quantile (monotone) coupling,
which is optimal in one dimension (exp_2 implementation_notes).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rho_phi(measure: str, alpha: float) -> float:
    """rho(Phi): the risk measure evaluated at the standard normal law.

    For VaR_alpha, rho(Phi) = Phi^{-1}(alpha) (Theorem 3.3).
    For TVaR_alpha, rho(Phi) = phi(Phi^{-1}(alpha)) / (1 - alpha)
    (Theorem 3.8 / Example 3.9, Section 3.3).
    """
    z = norm.ppf(alpha)
    if measure.upper() == "VAR":
        return float(z)
    if measure.upper() == "TVAR":
        return float(norm.pdf(z) / (1.0 - alpha))
    raise ValueError(f"Unknown risk measure: {measure}")


def empirical_var(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Empirical Value at Risk: the alpha-quantile of ``samples``.

    ``samples`` may be a 1D array (one VaR value) or a 2D array of shape
    (n_mc, n) / (n_mc,) batches, in which case the quantile is taken along
    the last axis for each Monte Carlo replication independently if 1D,
    otherwise a single scalar.
    """
    return np.quantile(samples, alpha)


def empirical_tvar(samples: np.ndarray, alpha: float) -> float:
    """Empirical Tail Value at Risk: mean of the values at or above the
    empirical alpha-quantile, TVaR_alpha(X) = E[X | X >= VaR_alpha(X)].
    """
    var = np.quantile(samples, alpha)
    tail = samples[samples >= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())


def empirical_risk_measure(samples: np.ndarray, measure: str, alpha: float) -> float:
    """Dispatch to :func:`empirical_var` or :func:`empirical_tvar`."""
    if measure.upper() == "VAR":
        return float(empirical_var(samples, alpha))
    if measure.upper() == "TVAR":
        return empirical_tvar(samples, alpha)
    raise ValueError(f"Unknown risk measure: {measure}")


def wasserstein_distance_1d(sample1: np.ndarray, sample2: np.ndarray, p: float = 1.0) -> float:
    """1D Wasserstein distance of order p via the quantile (monotone)
    coupling, which is optimal in one dimension (exp_2 implementation_notes).

    If the two samples have different sizes, both are mapped onto a common
    grid of quantile levels before the sorted values are compared.
    """
    s1 = np.sort(np.asarray(sample1, dtype=np.float64))
    s2 = np.sort(np.asarray(sample2, dtype=np.float64))
    if s1.size == s2.size:
        # Equal-size sorted samples already sit at the same quantile levels
        # (i - 0.5) / m, so the monotone coupling is just their pointwise
        # difference; re-gridding through np.quantile would reproduce the
        # same values at a much higher cost for the large n_mc used here.
        q1, q2 = s1, s2
    else:
        m = max(s1.size, s2.size)
        grid = (np.arange(1, m + 1) - 0.5) / m
        q1 = np.quantile(s1, grid)
        q2 = np.quantile(s2, grid)
    return float(np.mean(np.abs(q1 - q2) ** p) ** (1.0 / p))
