"""exp_3: Heterogeneous portfolio and stochastic cancellation (Section 4.3-4.4).

Reproduces, for a portfolio split into l risk segments with weights w_j,
standardised-loss scales sigma_j and (possibly segment-specific)
cancellation rates theta_j:

* Lemma 4.5 — the aggregated segment cancellation effect Delta_j^aggr is
  governed by the variance-weighted average cancellation rate
  theta_bar_sigma = sum_j pi_j theta_j, pi_j = w_j sigma_j^2 / <w, sigma^2>
  (eq. 4.2), checked against an empirical VaR_0.99 estimate obtained from
  independent Monte Carlo samples of the pre- and post-cancellation losses;
* Example 4.7 — the asymptotic ratio Delta_1^aggr / Delta_1 of the
  aggregated vs. standalone cancellation effect for segment 1, expected to
  converge to about 4.03 as the portfolio grows;
* Corollary 4.6 — the same ratio evaluated with the TVaR-type rho(Phi), to
  exercise the Theorem-3.8-style formula alongside the VaR-type one;
* the equal-cancellation-rate limit Delta_j^aggr / Delta_j -> sqrt(pi_j);
* a theory-only sensitivity sweep of the ratio as theta_1 moves relative to
  theta_bar_sigma, showing that aggregation helps (ratio > 1) exactly when
  theta_1 exceeds the portfolio average theta_bar_sigma;
* Lemma 4.8 — stochastic (Bernoulli) cancellation gives the same
  leading-order asymptotics as deterministic proportional cancellation,
  because Ntilde_j / n_j -> 1 - theta_j almost surely (strong law of large
  numbers).

Cancelling policies from a segment is a pathwise operation on that segment's
own realised losses (Sections 2-4), so each segment's pre- and
post-cancellation sub-portfolios are drawn with the common-random-numbers
coupling ``chunked_sample_sums_coupled`` (the post-cancellation sum and an
independent increment of the cancelled policies together reconstitute the
pre-cancellation sum), never as independent Monte Carlo draws and never as a
materialised ``(n_mc, n)`` array: independent sampling would add an O(n^{-1/2})
noise term from each side that swamps the O(n^{-1/2}) Lemma 4.5 signal itself.
"""

from __future__ import annotations

import argparse
import resource
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.heterogeneous import (
    segment_risk_contributions,
    effective_cancellation_rate,
    delta_j_aggr_theory,
    delta_j_standalone_theory,
    example_4_7_ratio,
    equal_rate_limit_ratio,
    bernoulli_cancellation_sizes,
)
from src.risk_measures import rho_phi, empirical_risk_measure
from data.generate_synthetic import sample_standardized, chunked_sample_sums_coupled

RESULTS_DIR = __file__.rsplit("/exp/", 1)[0] + "/results"
DIST_NAME = "exponential"


def simulate_segment_delta(
    weights: np.ndarray,
    sigmas: np.ndarray,
    thetas: np.ndarray,
    n: int,
    n_mc: int,
    rng: np.random.Generator,
    measure: str,
) -> Dict[str, np.ndarray]:
    """Empirical Delta_j^aggr for every segment j (paper step (g)).

    Each segment's post-cancellation sub-portfolio is coupled to its own
    pre-cancellation portfolio via ``chunked_sample_sums_coupled`` (cancelling
    policies removes them from the same realised segment, Section 4.3), so
    the aggregate pre- and post-cancellation losses share randomness
    segment-by-segment instead of being drawn independently. This removes a
    Monte Carlo noise source that otherwise swamps the O(n^{-1/2}) signal
    (Lemma 4.5) at the paper's small cancellation rates.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    l = weights.size

    def sampler(j: int):
        return lambda size, rng_: sigmas[j] * sample_standardized(DIST_NAME, size, rng_)

    n_j = np.round(weights * n).astype(int)
    ntilde_j = np.round(n_j * (1.0 - thetas)).astype(int)
    totals_pre = np.empty((l, n_mc), dtype=np.float64)
    totals_post = np.empty((l, n_mc), dtype=np.float64)
    for j in range(l):
        m_j = int(n_j[j] - ntilde_j[j])
        totals_pre[j], totals_post[j] = chunked_sample_sums_coupled(sampler(j), n_j[j], m_j, n_mc, rng)
    S_pre_total = totals_pre.sum(axis=0)
    RC_pre = segment_risk_contributions(weights, sigmas)
    S_post_total = totals_post.sum(axis=0)
    w_tilde = (1.0 - thetas) * weights
    w_tilde = w_tilde / w_tilde.sum()
    RC_post = segment_risk_contributions(w_tilde, sigmas)

    delta_j_empirical = np.empty(l, dtype=np.float64)
    for j in range(l):
        loss_pre_j = (RC_pre[j] / n_j[j]) * S_pre_total
        loss_post_j = (RC_post[j] / ntilde_j[j]) * S_post_total
        delta_j_empirical[j] = empirical_risk_measure(
            loss_post_j, measure, cfg.ALPHA
        ) - empirical_risk_measure(loss_pre_j, measure, cfg.ALPHA)

    return {
        "delta_j_empirical": delta_j_empirical,
        "n_j": n_j,
        "ntilde_j": ntilde_j,
        "RC_pre": RC_pre,
        "RC_post": RC_post,
        "totals_pre": totals_pre,
        "totals_post": totals_post,
    }


def standalone_delta_from_totals(
    totals_pre_j: np.ndarray, totals_post_j: np.ndarray, n_j: int, ntilde_j: int, measure: str
) -> float:
    """Empirical Delta_j for segment j simulated alone, as a homogeneous
    portfolio of size n_j with its own cancellation rate theta, reusing the
    segment's own pre-/post-cancellation totals already drawn for the
    aggregated computation (``simulate_segment_delta``).

    Segment j's standalone loss and its contribution to the aggregated
    portfolio are the same realised draws scaled differently (by 1/n_j vs.
    RC_j/n_j); applying the common-random-numbers principle used throughout
    this module (see the module docstring) to the standalone/aggregated
    comparison as well removes an extra independent Monte Carlo error term
    from the ratio Delta_j^aggr / Delta_j (Example 4.7), which otherwise adds
    noise comparable in size to the O(n^{-1/2}) signal being compared.
    """
    pre_mean = totals_pre_j / n_j
    post_mean = totals_post_j / ntilde_j
    return empirical_risk_measure(post_mean, measure, cfg.ALPHA) - empirical_risk_measure(
        pre_mean, measure, cfg.ALPHA
    )


def simulate_standalone_delta(
    sigma: float, theta: float, n_j: int, n_mc: int, rng: np.random.Generator, measure: str
) -> float:
    """Empirical Delta_j for segment j simulated alone, as a homogeneous
    portfolio of size n_j with its own cancellation rate theta, drawn with
    fresh (uncoupled) randomness. Used only where no matching aggregated
    totals are available; prefer :func:`standalone_delta_from_totals` when
    they are, for lower Monte Carlo variance."""

    def sampler(size, rng_):
        return sigma * sample_standardized(DIST_NAME, size, rng_)

    n_post = int(round(n_j * (1.0 - theta)))
    m = n_j - n_post
    pre_sum, post_sum = chunked_sample_sums_coupled(sampler, n_j, m, n_mc, rng)
    pre_mean = pre_sum / n_j
    post_mean = post_sum / n_post
    return empirical_risk_measure(post_mean, measure, cfg.ALPHA) - empirical_risk_measure(
        pre_mean, measure, cfg.ALPHA
    )


def _batched_segment_delta(
    weights: np.ndarray,
    sigmas: np.ndarray,
    thetas: np.ndarray,
    n: int,
    n_mc: int,
    n_batches: int,
    rng: np.random.Generator,
    measure: str,
    segment: int = 0,
) -> tuple:
    """Average Delta_segment^aggr and its common-random-numbers standalone
    counterpart over ``n_batches`` independent Monte Carlo batches of
    ``n_mc`` replications each.

    A single empirical VaR_0.99/TVaR_0.99 of ``n_mc`` replications is itself
    a noisy quantile/tail-mean estimator; at the paper's cancellation rates
    (theta down to 1%) the O(n^{-1/2}) signal of Lemma 4.5 / Example 4.7 can
    be of the same order as that quantile-estimation noise. Running
    ``n_batches`` independent batches and averaging the resulting deltas
    (not the per-batch ratios, which can be unstable when a single batch's
    denominator is near zero) reduces the variance of the reported
    quantities by a factor of ``n_batches`` while keeping the total number
    of simulated replications, and hence the per-batch memory footprint,
    unchanged.
    """
    delta_sum = 0.0
    standalone_sum = 0.0
    last_res: Dict[str, np.ndarray] = {}
    for _ in range(n_batches):
        res = simulate_segment_delta(weights, sigmas, thetas, n, n_mc, rng, measure)
        delta_sum += res["delta_j_empirical"][segment]
        standalone_sum += standalone_delta_from_totals(
            res["totals_pre"][segment],
            res["totals_post"][segment],
            res["n_j"][segment],
            res["ntilde_j"][segment],
            measure,
        )
        last_res = res
    return delta_sum / n_batches, standalone_sum / n_batches, last_res


def run_lemma45_and_example47(
    weights: np.ndarray,
    sigmas: np.ndarray,
    thetas: np.ndarray,
    n_grid: list,
    n_mc: int,
    n_batches: int,
    rng: np.random.Generator,
) -> tuple:
    """Steps 1-2: Lemma 4.5 check (segment 1) and Example 4.7 ratio, for
    every n in the grid, measure="VAR". Also exercises the TVaR-type
    (Corollary 4.6) formula at the largest n as a one-off comparison.

    Each (n, measure) point averages ``n_batches`` independent Monte Carlo
    batches of ``n_mc`` replications (see ``_batched_segment_delta``)."""
    rho_var = rho_phi("VAR", cfg.ALPHA)
    lemma45_rows = []
    example47_rows = []
    ratio_expected = cfg.EXP3_EX47_RATIO_EXPECTED

    for n in n_grid:
        delta1_empirical, delta1_standalone_empirical, res = _batched_segment_delta(
            weights, sigmas, thetas, n, n_mc, n_batches, rng, "VAR"
        )
        delta1_theory = delta_j_aggr_theory(0, weights, sigmas, thetas, n, rho_var)
        lemma45_rows.append(
            {"n": n, "segment": 1, "delta_empirical": delta1_empirical, "delta_theory": delta1_theory}
        )
        ratio_empirical = delta1_empirical / delta1_standalone_empirical
        ratio_theory = example_4_7_ratio(weights, sigmas, thetas)
        ratio_theory_direct = delta_j_aggr_theory(
            0, weights, sigmas, thetas, n, rho_var
        ) / delta_j_standalone_theory(0, weights, sigmas, thetas, n, rho_var)
        example47_rows.append(
            {
                "n": n,
                "measure": "VAR",
                "ratio_empirical": ratio_empirical,
                "ratio_theory": ratio_theory,
                "ratio_theory_direct": ratio_theory_direct,
                "ratio_expected": ratio_expected,
            }
        )
        print(
            f"[exp_3] Lemma 4.5 (VAR) n={n:>9}: delta1_empirical={delta1_empirical:.6f} "
            f"delta1_theory={delta1_theory:.6f} | Example 4.7 ratio_empirical={ratio_empirical:.3f} "
            f"ratio_theory={ratio_theory:.3f} (expected~{ratio_expected})"
        )

    # Corollary 4.6: exercise the TVaR-type rho(Phi) formula at the largest n.
    n_max = n_grid[-1]
    rho_tvar = rho_phi("TVAR", cfg.ALPHA)
    delta1_tvar_empirical, delta1_standalone_tvar_empirical, _ = _batched_segment_delta(
        weights, sigmas, thetas, n_max, n_mc, n_batches, rng, "TVAR"
    )
    ratio_empirical_tvar = delta1_tvar_empirical / delta1_standalone_tvar_empirical
    ratio_theory_tvar = (
        delta_j_aggr_theory(0, weights, sigmas, thetas, n_max, rho_tvar)
        / delta_j_standalone_theory(0, weights, sigmas, thetas, n_max, rho_tvar)
    )
    example47_rows.append(
        {
            "n": n_max,
            "measure": "TVAR",
            "ratio_empirical": ratio_empirical_tvar,
            "ratio_theory": ratio_theory_tvar,
            "ratio_theory_direct": ratio_theory_tvar,
            "ratio_expected": ratio_expected,
        }
    )
    print(
        f"[exp_3] Corollary 4.6 (TVAR) n={n_max:>9}: ratio_empirical={ratio_empirical_tvar:.3f} "
        f"ratio_theory={ratio_theory_tvar:.3f} (theory is measure-independent since rho(Phi) cancels)"
    )

    return pd.DataFrame(lemma45_rows), pd.DataFrame(example47_rows)


def run_equal_rate_case(n: int, n_mc: int, n_batches: int, rng: np.random.Generator) -> pd.DataFrame:
    """Step 3: equal cancellation-rate case, Delta_j^aggr / Delta_j -> sqrt(pi_j).

    Averages ``n_batches`` independent Monte Carlo batches per segment
    (see ``_batched_segment_delta``) to reduce the sampling noise of the
    VaR_0.99-based ratio."""
    weights = np.asarray(cfg.EXP3_EQUALRATE_WEIGHTS, dtype=np.float64)
    sigmas = np.asarray(cfg.EXP3_EQUALRATE_SIGMAS, dtype=np.float64)
    theta = cfg.EXP3_EQUALRATE_THETA
    thetas = np.full(weights.size, theta, dtype=np.float64)

    rows = []
    for j in range(weights.size):
        delta_empirical, delta_standalone, _ = _batched_segment_delta(
            weights, sigmas, thetas, n, n_mc, n_batches, rng, "VAR", segment=j
        )
        ratio_empirical = delta_empirical / delta_standalone
        ratio_theory = equal_rate_limit_ratio(j, weights, sigmas)
        rows.append(
            {"segment": j + 1, "ratio_empirical": ratio_empirical, "ratio_theory": ratio_theory}
        )
        print(
            f"[exp_3] Equal-rate segment {j + 1}: ratio_empirical={ratio_empirical:.3f} "
            f"ratio_theory=sqrt(pi_{j + 1})={ratio_theory:.3f}"
        )
    return pd.DataFrame(rows)


def run_sensitivity_sweep() -> pd.DataFrame:
    """Step 4: theory-only sensitivity sweep of the aggregated/standalone
    ratio as theta_1 varies relative to the fixed theta_2=0.10, holding the
    Example 4.7 weights/sigmas fixed. No Monte Carlo sampling is needed."""
    weights = np.asarray(cfg.EXP3_EX47_WEIGHTS, dtype=np.float64)
    sigmas = np.asarray(cfg.EXP3_EX47_SIGMAS, dtype=np.float64)
    theta_2 = cfg.EXP3_EX47_THETAS[1]
    rho_var = rho_phi("VAR", cfg.ALPHA)
    n_ref = 100_000

    rows = []
    for theta_1 in np.linspace(0.005, 0.5, 60):
        thetas = np.array([theta_1, theta_2])
        theta_bar = effective_cancellation_rate(weights, sigmas, thetas)
        ratio_theory = delta_j_aggr_theory(
            0, weights, sigmas, thetas, n_ref, rho_var
        ) / delta_j_standalone_theory(0, weights, sigmas, thetas, n_ref, rho_var)
        rows.append(
            {
                "theta_1": theta_1,
                "theta_1_over_theta_bar": theta_1 / theta_bar,
                "ratio_theory": ratio_theory,
            }
        )
    return pd.DataFrame(rows)


def plot_sensitivity(table: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["theta_1_over_theta_bar"], table["ratio_theory"], marker="o", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ratio = 1")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="theta_1 / theta_bar_sigma = 1")
    ax.set_xlabel("theta_1 / theta_bar_sigma")
    ax.set_ylabel("Delta_1^aggr / Delta_1 (theory)")
    ax.set_title("Example 4.7 sensitivity: aggregation helps iff theta_1 >= theta_bar_sigma")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_bernoulli_cancellation(n: int, n_mc: int, rng: np.random.Generator) -> pd.DataFrame:
    """Step 5: Lemma 4.8, stochastic (Bernoulli) cancellation vs the
    deterministic-theta theory.

    This is a *formula-based* verification: Ntilde_j is sampled directly via
    ``bernoulli_cancellation_sizes`` (no nested loss simulation), and for
    each replication we plug the *realised* cancellation fraction
    theta_realized_j = 1 - Ntilde_j / n_j into the closed-form
    ``delta_j_aggr_theory``. Averaging over replications estimates
    E[Delta_1^aggr(Ntilde)] under Bernoulli cancellation and checks that it
    matches the deterministic-theta value, consistent with Ntilde_j/n_j ->
    1 - theta_j almost surely (strong law of large numbers) so that the
    leading-order asymptotics coincide (Lemma 4.8).
    """
    weights = np.asarray(cfg.EXP3_EX47_WEIGHTS, dtype=np.float64)
    sigmas = np.asarray(cfg.EXP3_EX47_SIGMAS, dtype=np.float64)
    thetas = np.asarray(cfg.EXP3_EX47_THETAS, dtype=np.float64)
    rho_var = rho_phi("VAR", cfg.ALPHA)

    segment_sizes = np.round(weights * n).astype(int)
    remaining = bernoulli_cancellation_sizes(segment_sizes, thetas, n_mc, rng)
    theta_realized = 1.0 - remaining / segment_sizes[np.newaxis, :]

    deterministic_theory = delta_j_aggr_theory(0, weights, sigmas, thetas, n, rho_var)
    bernoulli_values = np.empty(n_mc, dtype=np.float64)
    for row in range(n_mc):
        bernoulli_values[row] = delta_j_aggr_theory(
            0, weights, sigmas, theta_realized[row], n, rho_var
        )
    bernoulli_mc_mean = float(bernoulli_values.mean())
    bernoulli_mc_std = float(bernoulli_values.std(ddof=1))
    relative_difference = (bernoulli_mc_mean - deterministic_theory) / deterministic_theory

    print(
        f"[exp_3] Lemma 4.8 n={n}: deterministic_theory={deterministic_theory:.6f} "
        f"bernoulli_mc_mean={bernoulli_mc_mean:.6f} (+/- {bernoulli_mc_std:.6f}) "
        f"relative_difference={relative_difference:.4%}"
    )

    return pd.DataFrame(
        [
            {
                "n": n,
                "deterministic_theory": deterministic_theory,
                "bernoulli_mc_mean": bernoulli_mc_mean,
                "bernoulli_mc_std": bernoulli_mc_std,
                "relative_difference": relative_difference,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_3: heterogeneous portfolio cancellation")
    parser.add_argument(
        "--full", action="store_true", help="use the paper-scale grid/n_mc instead of the reduced run config"
    )
    args = parser.parse_args()

    if args.full:
        n_grid = cfg.EXP3_N_GRID_PAPER
        n_mc = cfg.EXP3_N_MC_PAPER
        n_batches = cfg.EXP3_N_BATCHES_PAPER
        grid_label = "EXP3_N_GRID_PAPER"
        n_mc_label = "EXP3_N_MC_PAPER"
    else:
        n_grid = cfg.EXP3_N_GRID_RUN
        n_mc = cfg.EXP3_N_MC_RUN
        n_batches = cfg.EXP3_N_BATCHES_RUN
        grid_label = "EXP3_N_GRID_RUN"
        n_mc_label = "EXP3_N_MC_RUN"

    print("[exp_3] Realized configuration:")
    print(f"  mode            : {'--full (paper scale)' if args.full else 'reduced run (default)'}")
    print(f"  n_grid ({grid_label}) = {n_grid}")
    print(f"  n_mc   ({n_mc_label})   = {n_mc}, n_batches = {n_batches} (total MC draws per point = {n_mc * n_batches})")
    print(f"  ALPHA = {cfg.ALPHA}, RANDOM_SEED = {cfg.RANDOM_SEED}")
    print(f"  Example 4.7 weights/sigmas/thetas = {cfg.EXP3_EX47_WEIGHTS} / {cfg.EXP3_EX47_SIGMAS} / {cfg.EXP3_EX47_THETAS}")
    print(
        f"  Equal-rate weights/sigmas/theta = {cfg.EXP3_EQUALRATE_WEIGHTS} / "
        f"{cfg.EXP3_EQUALRATE_SIGMAS} / {cfg.EXP3_EQUALRATE_THETA}"
    )
    if not args.full:
        print(
            "  [exp_3] NOTE: using the reduced RUN grid/n_mc (not the paper-scale PAPER "
            "values) to stay well under the time/memory budget; pass --full for the paper scale."
        )

    rng = np.random.default_rng(cfg.RANDOM_SEED)

    weights = np.asarray(cfg.EXP3_EX47_WEIGHTS, dtype=np.float64)
    sigmas = np.asarray(cfg.EXP3_EX47_SIGMAS, dtype=np.float64)
    thetas = np.asarray(cfg.EXP3_EX47_THETAS, dtype=np.float64)

    # Steps 1-2: Lemma 4.5 check + Example 4.7 ratio (and a TVaR exercise).
    lemma45_table, example47_table = run_lemma45_and_example47(
        weights, sigmas, thetas, n_grid, n_mc, n_batches, rng
    )
    lemma45_table.to_csv(f"{RESULTS_DIR}/exp3_lemma45_check.csv", index=False)
    example47_table.to_csv(f"{RESULTS_DIR}/exp3_example47.csv", index=False)

    # Step 3: equal cancellation-rate limit.
    equalrate_table = run_equal_rate_case(n_grid[-1], n_mc, n_batches, rng)
    equalrate_table.to_csv(f"{RESULTS_DIR}/exp3_equalrate.csv", index=False)

    # Step 4: theory-only sensitivity sweep.
    sensitivity_table = run_sensitivity_sweep()
    sensitivity_table.to_csv(f"{RESULTS_DIR}/exp3_sensitivity.csv", index=False)
    plot_sensitivity(sensitivity_table, f"{RESULTS_DIR}/exp3_sensitivity.png")

    # Step 5: Bernoulli (stochastic) cancellation, Lemma 4.8.
    bernoulli_table = run_bernoulli_cancellation(n_grid[-1], n_mc, rng)
    bernoulli_table.to_csv(f"{RESULTS_DIR}/exp3_stochastic.csv", index=False)

    # Step 6: final summary.
    print("\n[exp_3] ===== Summary =====")
    print("[exp_3] Example 4.7 ratio convergence (VAR, segment 1 vs standalone):")
    print(example47_table[example47_table["measure"] == "VAR"].to_string(index=False))
    print("[exp_3] Corollary 4.6 TVaR exercise at largest n:")
    print(example47_table[example47_table["measure"] == "TVAR"].to_string(index=False))
    print("[exp_3] Equal cancellation-rate ratios (Delta_j^aggr / Delta_j -> sqrt(pi_j)):")
    print(equalrate_table.to_string(index=False))
    print("[exp_3] Bernoulli (Lemma 4.8) vs deterministic cancellation:")
    print(bernoulli_table.to_string(index=False))

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\n[exp_3] Peak memory usage: {peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()

