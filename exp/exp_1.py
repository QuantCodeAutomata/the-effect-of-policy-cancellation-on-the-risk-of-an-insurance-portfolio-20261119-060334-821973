"""exp_1: Constant cancellation regime (Section 3 of the paper).

Verifies, by Monte Carlo simulation, the asymptotic cancellation effect
Delta_{m,n}(X, rho) = rho_n(X) - rho_{n-m}(X) of removing a *fixed* number
``m`` of policies from a portfolio of ``n`` i.i.d. standardised losses:

* Theorem 3.3: for VaR_alpha, Delta_{m,n} ~ m*sigma*Phi^{-1}(alpha) / (2 n^{3/2})
  under Cramer's condition (finite exponential moments), fitted here against
  three such distributions (exponential, Student-t(5), Gamma(2,1)).
* Theorem 3.8: the same O(n^{-3/2}) rate holds for any L^1-Lipschitz
  (W_1-continuous) risk measure rho, with the Gaussian benchmark rho(Phi)
  (Example 3.9), here specialised to TVaR_alpha.
* The Cornish-Fisher expansion (eq. 3.2) is cross-checked against the direct
  empirical VaR of the simulated sample mean.
* Section 3.4 (Cramer's condition is necessary): a centred symmetric stable
  law with stability parameter alpha_hat < 2 violates Cramer's condition and
  Theorem 3.3 no longer applies; the cancellation effect instead decays at
  rate n^{1/alpha_hat - 1}, i.e. beta = 2 - 1/alpha_hat != 1.5, which this
  script verifies empirically as a counterexample.

All Monte Carlo replications use a *common random numbers* (CRN) coupling:
the sample of size n-m is obtained by removing the last m draws from the
very same n-draw realisation used for the sample of size n. This is
essential because the cancellation signal is O(n^{-3/2}), far smaller than
the O(n^{-1/2}) Monte Carlo noise that independent replications of size n
and n-m would contain.

Run with ``python -m exp.exp_1`` (reduced grid/replications, see
``src/config.py`` EXP1_*_RUN) or ``python -m exp.exp_1 --full`` (paper-scale
grid/replications, EXP1_*_PAPER); the latter is not time-budgeted here and
may take a long time, see module-level NOTE below.
"""

from __future__ import annotations

import argparse
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.config as cfg
from src.edgeworth import (
    cornish_fisher_quantile,
    delta_mn_var_theory,
    delta_mn_rho_theory,
    power_law_fit,
)
from src.risk_measures import rho_phi, empirical_risk_measure
from data.generate_synthetic import (
    sample_standardized,
    sample_stable,
    standardized_cumulants,
    chunked_sample_sums_with_tail,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MEASURES = ["VAR", "TVAR"]

# NOTE on the stable-law section's cost: scipy's `levy_stable.rvs` is about
# 30x slower per element than numpy's native generators (benchmarked: ~4.5M
# vs ~25-140M draws/sec), so reusing EXP1_N_MC_RUN unmodified for the stable
# counterexample would alone take on the order of 13 minutes, blowing the
# <10 minute budget for `python -m exp.exp_1`. We therefore cap the number of
# Monte Carlo replications used *only* in the stable section, *only* when
# --full is not given; src/config.py itself is never modified, and the same
# (n, m) grid is kept so the two sections remain directly comparable.
STABLE_N_MC_RUN_CAP = 2000

# NOTE on the stable-law section's memory: scipy's `levy_stable.rvs` (alpha !=
# 1, 2) uses an internal quadrature/interpolation algorithm that was measured
# to cost about 200 bytes per drawn element -- roughly 25x more than a plain
# float64 array (8 bytes) -- because it builds several full-sized internal
# working arrays per call. `chunked_sample_sums_with_tail`'s default
# `max_chunk_elements=20_000_000` was sized for ~160 MB chunks assuming 8
# bytes/element; for the stable sampler a single such chunk actually needs
# about 4 GiB, which alone would blow the shared 1.5 GiB budget. We therefore
# pass a much smaller chunk-size override for this sampler only (a pure
# streaming/chunking fix, not a reduction of n or n_mc).
STABLE_MAX_CHUNK_ELEMENTS = 1_500_000


@dataclass
class RunConfig:
    """Realised configuration for one invocation of this script."""

    full: bool
    distributions: list[str]
    n_grid: list[int]
    m_values: list[int]
    n_mc: int
    stable_alpha_hat: float
    stable_n_mc: int
    alpha: float = cfg.ALPHA
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"mode: {'FULL (paper-scale)' if self.full else 'RUN (reduced)'}",
            f"distributions: {self.distributions}",
            f"n_grid: {self.n_grid}",
            f"m_values: {self.m_values}",
            f"n_mc (Cramer section): {self.n_mc}",
            f"stable_alpha_hat: {self.stable_alpha_hat}",
            f"n_mc (stable section): {self.stable_n_mc}",
            f"alpha: {self.alpha}",
        ]
        for note in self.notes:
            lines.append(f"NOTE: {note}")
        return "\n".join(lines)


def get_run_config(full: bool) -> RunConfig:
    """Build the realised configuration, applying the stable-section Monte
    Carlo replication cap documented above when ``full`` is False."""
    if full:
        n_grid = list(cfg.EXP1_N_GRID_PAPER)
        n_mc = cfg.EXP1_N_MC_PAPER
        stable_n_mc = cfg.EXP1_N_MC_PAPER
        notes: list[str] = []
    else:
        n_grid = list(cfg.EXP1_N_GRID_RUN)
        n_mc = cfg.EXP1_N_MC_RUN
        stable_n_mc = min(cfg.EXP1_N_MC_RUN, STABLE_N_MC_RUN_CAP)
        notes = [
            f"stable-law section Monte Carlo replications capped at "
            f"{stable_n_mc} (instead of {n_mc}) to fit the time budget; "
            f"scipy.stats.levy_stable.rvs is ~30x slower per element than "
            f"numpy's native generators. The (n, m) grid is unchanged."
        ]
    return RunConfig(
        full=full,
        distributions=list(cfg.EXP1_DISTRIBUTIONS),
        n_grid=n_grid,
        m_values=list(cfg.EXP1_M_VALUES),
        n_mc=n_mc,
        stable_alpha_hat=cfg.EXP1_STABLE_ALPHA_HAT,
        stable_n_mc=stable_n_mc,
        notes=notes,
    )


def run_cramer_section(run_cfg: RunConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Theorem 3.3 / 3.8: Cramer-condition distributions.

    For each distribution and portfolio size n, draws n_mc coupled
    replications of the standardised sample mean of size n (and, via the
    shared tail, of size n-m for every m in EXP1_M_VALUES using the same
    underlying draws), and compares the empirical VaR/TVaR cancellation
    effect to its theoretical asymptotic.
    """
    max_m = max(run_cfg.m_values)
    rows: list[dict] = []
    for dist_name in run_cfg.distributions:
        kappa3, kappa4 = standardized_cumulants(dist_name)
        sampler = lambda size, rng_, _d=dist_name: sample_standardized(_d, size, rng_)
        for n in run_cfg.n_grid:
            assert max_m < n, f"m={max_m} must be < n={n}"
            totals, tail_values = chunked_sample_sums_with_tail(
                sampler, n=n, tail=max_m, n_mc=run_cfg.n_mc, rng=rng
            )
            mean_n = totals / n
            cf_var_n = cornish_fisher_quantile(run_cfg.alpha, n, kappa3, kappa4) / np.sqrt(n)
            rho_n = {measure: empirical_risk_measure(mean_n, measure, run_cfg.alpha) for measure in MEASURES}
            for m in run_cfg.m_values:
                tail_sum = tail_values[:, -m:].sum(axis=1)
                mean_nm = (totals - tail_sum) / (n - m)
                # Purely analytic check (no Monte Carlo noise): the Cornish-Fisher
                # quantile (eq. 3.2, Lemma 3.2) evaluated at n and at n-m, finite
                # differenced, should reduce to the closed-form Theorem 3.3 rate.
                # This verifies that the implemented CF expansion algebraically
                # produces the stated n^{-3/2} asymptotic, independently of the
                # Monte Carlo sampling noise discussed in results/RESULTS.md.
                cf_var_nm = cornish_fisher_quantile(run_cfg.alpha, n - m, kappa3, kappa4) / np.sqrt(n - m)
                delta_cf_analytic = cf_var_nm - cf_var_n
                for measure in MEASURES:
                    rho_nm = empirical_risk_measure(mean_nm, measure, run_cfg.alpha)
                    delta_empirical = rho_nm - rho_n[measure]
                    rho_phi_value = rho_phi(measure, run_cfg.alpha)
                    if measure == "VAR":
                        delta_theory = delta_mn_var_theory(m, n, sigma=1.0, alpha=run_cfg.alpha)
                    else:
                        delta_theory = delta_mn_rho_theory(m, n, sigma=1.0, rho_phi_value=rho_phi_value)
                    rows.append({
                        "dist": dist_name,
                        "n": n,
                        "m": m,
                        "measure": measure,
                        "rho_n": rho_n[measure],
                        "rho_nm": rho_nm,
                        "delta_empirical": delta_empirical,
                        "delta_theory": delta_theory,
                        "cf_var_n": cf_var_n if measure == "VAR" else np.nan,
                        "delta_cf_analytic": delta_cf_analytic if measure == "VAR" else np.nan,
                    })
    return pd.DataFrame(rows)


def fit_powerlaw(df: pd.DataFrame, group_cols: list[str], value_col: str = "delta_empirical") -> pd.DataFrame:
    """Fit ``value_col`` ~ C * n^{-beta} within each group of ``group_cols``."""
    fit_rows: list[dict] = []
    for key, group in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        group = group.sort_values("n")
        values = group[value_col].to_numpy()
        valid = np.isfinite(values) & (values != 0)
        c_coef, beta = power_law_fit(group["n"].to_numpy()[valid], np.abs(values[valid]))
        row = dict(zip(group_cols, key))
        row["C"] = c_coef
        row["beta"] = beta
        fit_rows.append(row)
    return pd.DataFrame(fit_rows)


def run_stable_section(run_cfg: RunConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, float]]:
    """Section 3.4 counterexample: a centred symmetric stable law with
    stability parameter alpha_hat < 2 violates Cramer's condition, so
    Theorem 3.3 does not apply; instead the cancellation effect decays like
    n^{1/alpha_hat - 1} rather than n^{-3/2} (beta = 2 - 1/alpha_hat).
    """
    alpha_hat = run_cfg.stable_alpha_hat
    max_m = max(run_cfg.m_values)

    # One-shot direct Monte Carlo estimate of VaR/TVaR of the single-policy
    # loss X itself (rho_X), used as the scale factor of the theoretical rate.
    raw_sample = sample_stable(alpha_hat, run_cfg.n_mc, rng)
    rho_x = {measure: empirical_risk_measure(raw_sample, measure, run_cfg.alpha) for measure in MEASURES}

    sampler = lambda size, rng_: sample_stable(alpha_hat, size, rng_)
    rows: list[dict] = []
    for n in run_cfg.n_grid:
        assert max_m < n, f"m={max_m} must be < n={n}"
        totals, tail_values = chunked_sample_sums_with_tail(
            sampler, n=n, tail=max_m, n_mc=run_cfg.stable_n_mc, rng=rng,
            max_chunk_elements=STABLE_MAX_CHUNK_ELEMENTS,
        )
        mean_n = totals / n
        rho_n = {measure: empirical_risk_measure(mean_n, measure, run_cfg.alpha) for measure in MEASURES}
        for m in run_cfg.m_values:
            tail_sum = tail_values[:, -m:].sum(axis=1)
            mean_nm = (totals - tail_sum) / (n - m)
            for measure in MEASURES:
                rho_nm = empirical_risk_measure(mean_nm, measure, run_cfg.alpha)
                delta_empirical = rho_nm - rho_n[measure]
                delta_theory_stable = (
                    (n - m) ** (1.0 / alpha_hat - 1.0) - n ** (1.0 / alpha_hat - 1.0)
                ) * rho_x[measure]
                rows.append({
                    "n": n,
                    "m": m,
                    "measure": measure,
                    "rho_n": rho_n[measure],
                    "rho_nm": rho_nm,
                    "delta_empirical": delta_empirical,
                    "delta_theory_stable": delta_theory_stable,
                    "rho_X": rho_x[measure],
                })
    return pd.DataFrame(rows), rho_x


def make_plot(
    df_cramer: pd.DataFrame,
    fit_cramer: pd.DataFrame,
    df_stable: pd.DataFrame,
    fit_stable: pd.DataFrame,
    path: Path,
) -> None:
    """Figure: log|Delta_empirical| vs log(n) at m=1, one subplot per risk
    measure, one line per Cramer-condition distribution plus the stable
    counterexample, fitted power-law slope beta in the legend."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for ax, measure in zip(axes, MEASURES):
        for dist_name, group in df_cramer[(df_cramer["m"] == 1) & (df_cramer["measure"] == measure)].groupby("dist"):
            group = group.sort_values("n")
            beta_row = fit_cramer[(fit_cramer["dist"] == dist_name) & (fit_cramer["m"] == 1) & (fit_cramer["measure"] == measure)]
            beta = float(beta_row["beta"].iloc[0]) if not beta_row.empty else float("nan")
            ax.plot(
                group["n"], np.abs(group["delta_empirical"]), marker="o", markersize=4,
                label=f"{dist_name} (beta={beta:.2f})",
            )
        stable_group = df_stable[(df_stable["m"] == 1) & (df_stable["measure"] == measure)].sort_values("n")
        beta_row = fit_stable[(fit_stable["m"] == 1) & (fit_stable["measure"] == measure)]
        beta = float(beta_row["beta"].iloc[0]) if not beta_row.empty else float("nan")
        ax.plot(
            stable_group["n"], np.abs(stable_group["delta_empirical"]), marker="s", markersize=4, linestyle="--",
            label=f"stable alpha_hat (beta={beta:.2f})",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel(f"|Delta_(1,n)| ({measure}, log scale)")
        ax.set_title(f"Cancellation effect decay, m=1, {measure}")
        ax.legend(fontsize=7)
    fig.suptitle("exp_1: power-law decay of the cancellation effect (Theorem 3.3 / 3.8 vs Section 3.4 counterexample)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Use paper-scale grid/replications (EXP1_*_PAPER) instead of the reduced EXP1_*_RUN values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = get_run_config(args.full)
    print("[exp_1] realized configuration:")
    print(run_cfg.describe())

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[exp_1] running Cramer-condition section (Theorem 3.3 / 3.8)...")
    df_cramer = run_cramer_section(run_cfg, rng)
    df_cramer.to_csv(RESULTS_DIR / "exp1_delta_cramer.csv", index=False)
    fit_cramer = fit_powerlaw(df_cramer, ["dist", "m", "measure"])
    fit_cramer["beta_minus_1p5"] = fit_cramer["beta"] - 1.5
    fit_cramer.to_csv(RESULTS_DIR / "exp1_powerlaw_fit.csv", index=False)
    # Analytic (Monte-Carlo-free) finite-difference check of the Cornish-Fisher
    # quantile, restricted to measure=VAR rows (see run_cramer_section).
    cf_rows = df_cramer[(df_cramer["measure"] == "VAR")]
    fit_cf = fit_powerlaw(cf_rows, ["dist", "m"], value_col="delta_cf_analytic")
    fit_cf["beta_minus_1p5"] = fit_cf["beta"] - 1.5
    fit_cf.to_csv(RESULTS_DIR / "exp1_powerlaw_fit_cf_analytic.csv", index=False)
    print(f"[exp_1] Cramer section done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("[exp_1] running stable-law counterexample section (Section 3.4)...")
    df_stable, rho_x = run_stable_section(run_cfg, rng)
    df_stable.to_csv(RESULTS_DIR / "exp1_stable_counterexample.csv", index=False)
    fit_stable = fit_powerlaw(df_stable, ["m", "measure"])
    fit_stable["beta_expected"] = 2.0 - 1.0 / run_cfg.stable_alpha_hat
    fit_stable.to_csv(RESULTS_DIR / "exp1_powerlaw_fit_stable.csv", index=False)
    print(f"[exp_1] stable section done in {time.time() - t1:.1f}s")

    make_plot(df_cramer, fit_cramer, df_stable, fit_stable, RESULTS_DIR / "exp1_powerlaw.png")

    print("\n[exp_1] summary of fitted power-law exponents beta (expected 1.5 for Cramer, "
          f"{2.0 - 1.0 / run_cfg.stable_alpha_hat:.4f} for the stable counterexample):")
    for _, row in fit_cramer.iterrows():
        print(f"  dist={row['dist']:<12} m={int(row['m']):<3} measure={row['measure']:<4} "
              f"beta={row['beta']:.4f}  (beta-1.5={row['beta_minus_1p5']:+.4f})")
    for _, row in fit_stable.iterrows():
        print(f"  stable        m={int(row['m']):<3} measure={row['measure']:<4} "
              f"beta={row['beta']:.4f}  (beta-expected={row['beta'] - row['beta_expected']:+.4f})")
    print(f"  stable rho_X: VAR={rho_x['VAR']:.4f}  TVAR={rho_x['TVAR']:.4f}")
    print("\n[exp_1] analytic (Monte-Carlo-free) Cornish-Fisher finite-difference check, VAR only:")
    for _, row in fit_cf.iterrows():
        print(f"  dist={row['dist']:<12} m={int(row['m']):<3} "
              f"beta={row['beta']:.4f}  (beta-1.5={row['beta_minus_1p5']:+.4f})")

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[exp_1] total wall time: {time.time() - t0:.1f}s")
    print(f"[exp_1] peak memory (ru_maxrss): {peak_kib / 1024.0:.1f} MiB")


if __name__ == "__main__":
    main()
