"""Synthetic i.i.d. loss samples for the Monte Carlo experiments.

All experiments in this paper ("The effect of policy cancellation on the risk
of an insurance portfolio") are evaluated on synthetic loss distributions:
the paper states only the theoretical conditions (Cramer's condition, finite
moments) and leaves the specific distributions to the experimenter
(exp_1/exp_2/exp_3 ``uncertainties``). This module standardises each
candidate distribution to mean 0, variance 1 and draws i.i.d. samples from
it, and provides a memory-bounded routine to accumulate sums of very large
samples without ever holding a full ``n_mc x n`` array in memory.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

# Maximum number of floating point numbers materialised at once when
# accumulating chunked sums (about 150 MB at 8 bytes per float), keeping a
# single process well under the shared 8 GiB memory budget.
MAX_CHUNK_ELEMENTS = 20_000_000


def _standardization_constants(dist_name: str) -> tuple[float, float]:
    """Return (mean, std) of the raw (unstandardised) distribution."""
    if dist_name == "exponential":
        d = stats.expon(scale=1.0)
    elif dist_name == "student_t5":
        d = stats.t(df=5)
    elif dist_name == "gamma21":
        d = stats.gamma(a=2.0, scale=1.0)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    return float(d.mean()), float(d.std())


def sample_standardized(dist_name: str, size: tuple | int, rng: np.random.Generator) -> np.ndarray:
    """Draw i.i.d. samples from ``dist_name``, standardised to mean 0, variance 1.

    Supported names: "exponential" (Exponential(1)), "student_t5" (Student-t
    with 5 degrees of freedom), "gamma21" (Gamma(2,1)). These satisfy
    Cramer's condition and have finite fourth moment (exp_1 data_requirements).
    """
    mean, std = _standardization_constants(dist_name)
    if dist_name == "exponential":
        raw = rng.exponential(scale=1.0, size=size)
    elif dist_name == "student_t5":
        raw = rng.standard_t(df=5, size=size)
    elif dist_name == "gamma21":
        raw = rng.gamma(shape=2.0, scale=1.0, size=size)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    return (raw - mean) / std


def sample_stable(alpha_hat: float, size: tuple | int, rng: np.random.Generator) -> np.ndarray:
    """Draw i.i.d. samples from a centred symmetric stable law with stability
    parameter ``alpha_hat`` (beta=0), used as the Cramer-condition counterexample
    (exp_1 implementation_notes: scipy.stats.levy_stable with alpha=alpha_hat, beta=0).

    The stable law with alpha_hat < 2 has infinite variance, so "standardised"
    here means scale=1 under scipy's parameterisation, not unit variance.
    """
    return stats.levy_stable.rvs(alpha=alpha_hat, beta=0.0, size=size, random_state=rng)


def standardized_cumulants(dist_name: str) -> tuple[float, float]:
    """Third and fourth cumulants (kappa_3, kappa_4) of the standardised
    distribution, computed from exact raw moments via the moment-cumulant
    relations kappa_3 = mu_3, kappa_4 = mu_4 - 3 (implementation_notes), using
    scipy's closed-form raw moments of the unstandardised law.
    """
    if dist_name == "exponential":
        d = stats.expon(scale=1.0)
    elif dist_name == "student_t5":
        d = stats.t(df=5)
    elif dist_name == "gamma21":
        d = stats.gamma(a=2.0, scale=1.0)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    mean, std = float(d.mean()), float(d.std())
    # Standardised third and fourth central moments via scipy's stats() call,
    # which returns (mean, variance, skewness, excess kurtosis) analytically.
    _, _, skew, exkurt = d.stats(moments="mvsk")
    kappa3 = float(skew)  # kappa_3 = mu_3 for a standardised variable = skewness
    kappa4 = float(exkurt)  # kappa_4 = mu_4 - 3 = excess kurtosis
    return kappa3, kappa4


def chunked_sample_sums(
    sampler: Callable[[tuple, np.random.Generator], np.ndarray],
    n: int,
    n_mc: int,
    rng: np.random.Generator,
    max_chunk_elements: int = MAX_CHUNK_ELEMENTS,
) -> np.ndarray:
    """Sum of ``n`` i.i.d. draws for each of ``n_mc`` independent Monte Carlo
    replications, without ever materialising the full ``(n_mc, n)`` array.

    ``sampler(size, rng)`` must draw ``size`` i.i.d. standardised variates.
    Returns an array of shape ``(n_mc,)`` with the replication sums.
    """
    totals = np.zeros(n_mc, dtype=np.float64)
    remaining = n
    chunk_n = max(1, max_chunk_elements // max(n_mc, 1))
    while remaining > 0:
        c = min(chunk_n, remaining)
        batch = sampler((n_mc, c), rng)
        totals += batch.sum(axis=1)
        remaining -= c
    return totals


def chunked_sample_sums_with_tail(
    sampler: Callable[[tuple, np.random.Generator], np.ndarray],
    n: int,
    tail: int,
    n_mc: int,
    rng: np.random.Generator,
    max_chunk_elements: int = MAX_CHUNK_ELEMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`chunked_sample_sums`, but additionally returns the last
    ``tail`` draws of each replication (shape ``(n_mc, tail)``), so that sums
    for several fixed cancellation counts ``m <= tail`` can be derived from a
    single pass by subtracting partial sums of the tail (used by exp_1).
    """
    totals = np.zeros(n_mc, dtype=np.float64)
    tail_values = np.zeros((n_mc, tail), dtype=np.float64)
    remaining = n
    chunk_n = max(1, max_chunk_elements // max(n_mc, 1))
    while remaining > 0:
        c = min(chunk_n, remaining)
        batch = sampler((n_mc, c), rng)
        totals += batch.sum(axis=1)
        # Keep only the most recent `tail` columns seen so far.
        tail_values = np.concatenate([tail_values, batch], axis=1)[:, -tail:]
        remaining -= c
    return totals, tail_values
