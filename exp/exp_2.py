"""exp_2: Proportional cancellation in a homogeneous portfolio (Section 4.1-4.2).

Reproduces, for a homogeneous portfolio of i.i.d. standardized losses under a
proportional cancellation regime (a fixed fraction theta of the n policies is
cancelled, m = floor(theta * n)):

* Delta_{theta,n}(X, rho) and CCR_{theta,n}(Phi, rho), empirical vs. the
  asymptotic formulas of Lemma 4.1 / Theorem 4.3, and the small-theta
  approximation (eq. 1.1), for VaR_0.99 and TVaR_0.99;
* the fitted decay rate beta of |Delta_{theta,n}| ~ C * n^{-beta} (expected
  beta ~ 1/2 from Lemma 4.1 / Theorem 4.3);
* the lambda_theta / theta curve of Section 2.2, Example 2.5;
* a one-dimensional Wasserstein-distance CLT check (Lemma 4.2): the
  normalized portfolio mean converges to the standard normal law at rate
  n^{-1/2} in Wasserstein-1/2 distance;
* a check that the remainder n * (Delta_empirical - Delta_theory) stays
  bounded (consistent with an O(n^{-1}) second-order term after the leading
  n^{-1/2} term, under a finite (p+2)-th moment / L^p-Lipschitz risk
  measure).

Memory note on sampling m close to n (theta up to 0.90):
  Unlike exp_1 (fixed small m), here m = floor(theta * n) can be a large
  fraction of n (up to 90%), so storing the "tail" of m individual losses
  per Monte Carlo replication (``chunked_sample_sums_with_tail``) would
  require an (n_mc, m) array -- for m close to n and n_mc in the tens of
  thousands this blows past the shared 1.5 GiB budget. Instead we draw the
  size-n portfolio and the size-(n-m) remaining portfolio as two
  INDEPENDENT Monte Carlo samples via ``chunked_sample_sums``, which only
  ever materializes O(n_mc) totals regardless of n or m. This is a
  deliberate, documented modeling choice: the paper's Delta_{theta,n}
  formula is a deterministic asymptotic limit (not a coupled, pathwise
  statement), so an independent-sample Monte Carlo estimator of
  rho(mean_{n-m}) - rho(mean_n) is valid -- just somewhat noisier than a
  coupled estimator would be -- while staying within the memory budget.
"""

from __future__ import annotations

import argparse
import resource
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.config as cfg
from src.proportional import (
    lambda_theta,
    delta_theta_n_theory,
    ccr_theta_n_theory,
    ccr_small_theta_approximation,
    lambda_theta_max_location,
)
from src.risk_measures import rho_phi, empirical_risk_measure, wasserstein_distance_1d
from src.edgeworth import power_law_fit
from data.generate_synthetic import sample_standardized, chunked_sample_sums, chunked_sample_sums_coupled

RESULTS_DIR = __file__.rsplit("/exp/", 1)[0] + "/results"
MEASURES = ["VAR", "TVAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp_2: proportional cancellation (Section 4.1-4.2)")
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper-scale grid/n_mc (EXP2_*_PAPER) instead of the reduced run config",
    )
    return parser.parse_args()


def make_sampler(dist_name: str) -> Callable[[tuple, np.random.Generator], np.ndarray]:
    """Bind ``dist_name`` into a ``sampler(size, rng)`` closure, as expected
    by ``chunked_sample_sums``."""

    def sampler(size: tuple, rng: np.random.Generator) -> np.ndarray:
        return sample_standardized(dist_name, size, rng)

    return sampler


def step1_delta_ccr(
    dist_names: list[str],
    n_grid: list[int],
    theta_grid: list[float],
    n_mc: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Lemma 4.1 / Theorem 4.3: empirical vs. theoretical Delta_{theta,n} and
    CCR_{theta,n}, plus the small-theta approximation (eq. 1.1)."""
    rows = []
    for dist_name in dist_names:
        sampler = make_sampler(dist_name)
        for n in n_grid:
            for theta in theta_grid:
                m = int(np.floor(theta * n))
                if m < 1 or m >= n:
                    print(f"[exp_2] step1: skip dist={dist_name} n={n} theta={theta} (m={m} out of range)")
                    continue
                # Coupled draw: the (n-m)-portfolio is the literal
                # sub-portfolio retained after cancelling m of the n
                # policies (see chunked_sample_sums_coupled docstring),
                # which removes the independent-sampling Monte Carlo noise
                # that would otherwise swamp the O(n^{-1/2}) signal.
                totals_n, totals_nm = chunked_sample_sums_coupled(sampler, n, m, n_mc, rng)
                mean_n = totals_n / n
                mean_nm = totals_nm / (n - m)
                for measure in MEASURES:
                    rho_n = empirical_risk_measure(mean_n, measure, alpha)
                    rho_nm = empirical_risk_measure(mean_nm, measure, alpha)
                    delta_empirical = rho_nm - rho_n
                    rho_phi_value = rho_phi(measure, alpha)
                    delta_theory = delta_theta_n_theory(theta, n, sigma=1.0, rho_phi_value=rho_phi_value)
                    ccr_empirical = ((n - m) / m) * delta_empirical
                    ccr_theory = ccr_theta_n_theory(theta, n, rho_phi_value)
                    ccr_small_theta = ccr_small_theta_approximation(theta, n, sigma=1.0, rho_phi_value=rho_phi_value)
                    rows.append({
                        "dist": dist_name,
                        "n": n,
                        "theta": theta,
                        "m": m,
                        "measure": measure,
                        "delta_empirical": delta_empirical,
                        "delta_theory": delta_theory,
                        "ccr_empirical": ccr_empirical,
                        "ccr_theory": ccr_theory,
                        "ccr_small_theta": ccr_small_theta,
                    })
    return pd.DataFrame(rows)


def step1_powerlaw_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Fit |Delta_empirical| ~ C * n^{-beta} for each (dist, theta, measure)."""
    rows = []
    for (dist_name, theta, measure), group in df.groupby(["dist", "theta", "measure"]):
        group = group.sort_values("n")
        c_fit, beta_fit = power_law_fit(group["n"].to_numpy(), group["delta_empirical"].abs().to_numpy())
        rows.append({
            "dist": dist_name,
            "theta": theta,
            "measure": measure,
            "C": c_fit,
            "beta": beta_fit,
            "beta_minus_0p5": beta_fit - 0.5,
        })
    return pd.DataFrame(rows)


def step2_lambda_theta_curve() -> tuple[pd.DataFrame, float, float]:
    """Section 2.2, Example 2.5: lambda_theta / theta curve.

    Note: the argmax of the ratio lambda_theta/theta need not coincide with
    the argmax of lambda_theta itself (reported by
    ``lambda_theta_max_location``, theta* = 3/4); both are reported here.
    """
    theta_fine = np.linspace(0.001, 0.999, 500)
    ratio = lambda_theta(theta_fine) / theta_fine
    idx = int(np.argmax(ratio))
    theta_ratio_argmax, ratio_argmax = float(theta_fine[idx]), float(ratio[idx])

    df = pd.DataFrame({"theta": theta_fine, "lambda_over_theta": ratio})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta_fine, ratio, label=r"$\lambda_\theta / \theta$")
    ax.axvline(0.75, color="red", linestyle="--", label=r"$\theta = 0.75$ (argmax of $\lambda_\theta$)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\lambda_\theta / \theta$")
    ax.set_title("Section 2.2 (Example 2.5): lambda_theta / theta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_lambda_theta.png", dpi=150)
    plt.close(fig)

    df.to_csv(f"{RESULTS_DIR}/exp2_lambda_theta.csv", index=False)
    return df, theta_ratio_argmax, ratio_argmax


def step3_print_small_theta_check(df: pd.DataFrame, n_grid: list[int]) -> None:
    """Print ccr_empirical / ccr_theory vs. ccr_small_theta (relative error)
    for the smallest thetas, at the two largest n in the grid."""
    small_thetas = [0.01, 0.05, 0.10]
    largest_n = sorted(n_grid)[-2:]
    subset = df[df["theta"].isin(small_thetas) & df["n"].isin(largest_n)].copy()
    if subset.empty:
        print("[exp_2] step3: no rows available for the small-theta check (grid too short)")
        return
    subset["rel_err_theory_vs_small"] = (subset["ccr_theory"] - subset["ccr_small_theta"]) / subset["ccr_small_theta"]
    subset["rel_err_empirical_vs_small"] = (
        subset["ccr_empirical"] - subset["ccr_small_theta"]
    ) / subset["ccr_small_theta"]
    cols = [
        "dist", "n", "theta", "measure", "ccr_empirical", "ccr_theory", "ccr_small_theta",
        "rel_err_theory_vs_small", "rel_err_empirical_vs_small",
    ]
    print("[exp_2] step3: small-theta approximation check (largest n in grid):")
    print(subset[cols].sort_values(["dist", "measure", "n", "theta"]).to_string(index=False))


def step4_wasserstein(
    dist_name: str, n_grid: list[int], n_mc: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lemma 4.2: Wasserstein-distance CLT check for sqrt(n) * mean_n vs. the
    standard normal law, for orders p=1 and p=2."""
    sampler = make_sampler(dist_name)
    rows = []
    for n in n_grid:
        totals_n = chunked_sample_sums(sampler, n, n_mc, rng)
        z_n_samples = np.sqrt(n) * totals_n / n
        # Lemma 4.2 concerns the law of sqrt(n)*X_bar_n vs. the population law
        # Phi, not two independent finite samples. Using the exact quantile
        # function of Phi as the reference (rather than a second Monte Carlo
        # draw from N(0,1)) removes an extra O(n_mc^{-1/2}) sampling-noise
        # floor that otherwise swamps the n^{-1/2} Wasserstein CLT rate.
        quantile_grid = (np.arange(1, n_mc + 1) - 0.5) / n_mc
        normal_quantiles = norm.ppf(quantile_grid)
        w1 = wasserstein_distance_1d(z_n_samples, normal_quantiles, p=1.0)
        w2 = wasserstein_distance_1d(z_n_samples, normal_quantiles, p=2.0)
        rows.append({"n": n, "w1": w1, "w2": w2})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/exp2_wasserstein.csv", index=False)

    n_arr = df["n"].to_numpy()
    c1, beta1 = power_law_fit(n_arr, df["w1"].to_numpy())
    c2, beta2 = power_law_fit(n_arr, df["w2"].to_numpy())
    fit_df = pd.DataFrame([
        {"order": "w1", "C": c1, "beta": beta1, "beta_minus_0p5": beta1 - 0.5},
        {"order": "w2", "C": c2, "beta": beta2, "beta_minus_0p5": beta2 - 0.5},
    ])
    fit_df.to_csv(f"{RESULTS_DIR}/exp2_wasserstein_fit.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n_arr, df["w1"], "o-", label=f"W1 (fitted slope={-beta1:.3f})")
    ax.loglog(n_arr, df["w2"], "s-", label=f"W2 (fitted slope={-beta2:.3f})")
    ax.set_xlabel("n")
    ax.set_ylabel("Wasserstein distance to N(0,1)")
    ax.set_title(f"Lemma 4.2: Wasserstein CLT check ({dist_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_wasserstein.png", dpi=150)
    plt.close(fig)
    return df, fit_df


def step5_sharpened_remainder(df: pd.DataFrame, dist_name: str, theta: float, measure: str) -> pd.DataFrame:
    """Check that n * (Delta_empirical - Delta_theory) stays bounded, i.e.
    the remainder after the n^{-1/2} leading term decays like O(n^{-1})."""
    subset = df[(df["dist"] == dist_name) & (df["theta"] == theta) & (df["measure"] == measure)].copy()
    subset = subset.sort_values("n")
    subset["residual_times_n"] = subset["n"] * (subset["delta_empirical"] - subset["delta_theory"])
    out = subset[["n", "residual_times_n"]].reset_index(drop=True)
    out.to_csv(f"{RESULTS_DIR}/exp2_sharpened_remainder.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out["n"], out["residual_times_n"], "o-")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$n \cdot (\Delta_{empirical} - \Delta_{theory})$")
    ax.set_title(f"Sharpened remainder check ({dist_name}, theta={theta}, {measure})")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_sharpened_remainder.png", dpi=150)
    plt.close(fig)
    return out


def step6_combined_powerlaw_plot(df: pd.DataFrame, fits_df: pd.DataFrame, theta: float, dist_names: list[str]) -> None:
    """One figure, two subplots (VAR, TVAR): log|Delta_empirical| vs log(n)
    for theta=0.10 across all distributions, with fitted slopes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, measure in zip(axes, MEASURES):
        for dist_name in dist_names:
            subset = df[(df["dist"] == dist_name) & (df["theta"] == theta) & (df["measure"] == measure)]
            subset = subset.sort_values("n")
            fit_row = fits_df[
                (fits_df["dist"] == dist_name) & (fits_df["theta"] == theta) & (fits_df["measure"] == measure)
            ]
            beta_fit = float(fit_row["beta"].iloc[0]) if not fit_row.empty else float("nan")
            ax.loglog(
                subset["n"], subset["delta_empirical"].abs(), "o-",
                label=f"{dist_name} (slope={-beta_fit:.3f})",
            )
        ax.set_xlabel("n")
        ax.set_ylabel(r"$|\Delta_{empirical}|$")
        ax.set_title(f"{measure}_0.99, theta={theta}")
        ax.legend(fontsize=8)
    fig.suptitle("Power-law decay of |Delta_{theta,n}| (Lemma 4.1 / Theorem 4.3)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_powerlaw.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dist_names = cfg.EXP2_DISTRIBUTIONS
    theta_grid = cfg.EXP2_THETA_GRID
    if args.full:
        n_grid = cfg.EXP2_N_GRID_PAPER
        n_mc = cfg.EXP2_N_MC_PAPER
    else:
        n_grid = cfg.EXP2_N_GRID_RUN
        n_mc = cfg.EXP2_N_MC_RUN
        # Runtime safety margin: a dry run with the full cfg.EXP2_N_MC_RUN
        # (8000) took ~9.7 minutes on this machine, which leaves almost no
        # buffer under the 10-minute budget on a host that also shares CPU
        # with other concurrent processes. We therefore cap n_mc for the
        # reduced (non --full) run, keeping the full n/theta grids intact
        # (they matter for the power-law fits and the small-theta check);
        # only the Monte Carlo replication count is reduced, which trades a
        # little extra noise for a safe runtime margin. Use --full for the
        # unreduced paper-scale n_mc.
        n_mc_cap = 5_000
        if n_mc > n_mc_cap:
            print(f"[exp_2] runtime safety reduction: n_mc capped at {n_mc_cap} (cfg.EXP2_N_MC_RUN = {n_mc})")
            n_mc = n_mc_cap
    alpha = cfg.ALPHA

    print("[exp_2] realized configuration:")
    print(f"  mode            = {'--full (paper-scale)' if args.full else 'reduced run config'}")
    print(f"  distributions   = {dist_names}")
    print(f"  theta grid      = {theta_grid}")
    print(f"  n grid          = {n_grid}")
    print(f"  n_mc            = {n_mc}")
    print(f"  alpha           = {alpha}")

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    t0 = time.time()

    # --- Step 1: Delta_{theta,n} / CCR_{theta,n}, empirical vs. theory -----
    df = step1_delta_ccr(dist_names, n_grid, theta_grid, n_mc, alpha, rng)
    df.to_csv(f"{RESULTS_DIR}/exp2_delta_proportional.csv", index=False)
    fits_df = step1_powerlaw_fits(df)
    fits_df.to_csv(f"{RESULTS_DIR}/exp2_powerlaw_fit.csv", index=False)
    print(f"[exp_2] step1 done in {time.time() - t0:.1f}s ({len(df)} rows)")

    # --- Step 2: lambda_theta / theta curve ---------------------------------
    _, theta_ratio_argmax, ratio_argmax = step2_lambda_theta_curve()
    theta_lambda_star, lambda_star = lambda_theta_max_location()
    print(
        f"[exp_2] step2: argmax(lambda_theta) at theta={theta_lambda_star} "
        f"(lambda={lambda_star}); argmax(lambda_theta/theta) at "
        f"theta={theta_ratio_argmax:.4f} (ratio={ratio_argmax:.4f})"
    )

    # --- Step 3: small-theta approximation check (stdout only) --------------
    step3_print_small_theta_check(df, n_grid)

    # --- Step 4: Wasserstein CLT check (Lemma 4.2) ---------------------------
    # Uses its own, much larger Monte Carlo size and its own n-grid (kept
    # well below that MC size): the diagnostic estimates W_p(Law(sqrt(n)*
    # X_bar_n), Phi) from a finite n_mc-sample, which carries its own
    # O(n_mc^{-1/2}) sampling-noise floor that would otherwise dominate the
    # O(n^{-1/2}) CLT signal at the larger n values of n_grid/EXP2_N_MC_RUN.
    t1 = time.time()
    wass_dist_name = dist_names[0]
    if args.full:
        n_mc_wass, n_grid_wass = cfg.EXP2_N_MC_WASSERSTEIN_PAPER, cfg.EXP2_N_GRID_WASSERSTEIN_PAPER
    else:
        n_mc_wass, n_grid_wass = cfg.EXP2_N_MC_WASSERSTEIN_RUN, cfg.EXP2_N_GRID_WASSERSTEIN_RUN
    wass_df, wass_fit_df = step4_wasserstein(wass_dist_name, n_grid_wass, n_mc_wass, rng)
    print(f"[exp_2] step4 done in {time.time() - t1:.1f}s (n_mc={n_mc_wass}, n_grid={n_grid_wass})")

    # --- Step 5: sharpened remainder check ------------------------------------
    sharp_dist_name, sharp_theta, sharp_measure = dist_names[0], 0.10, "VAR"
    if sharp_theta in theta_grid:
        step5_sharpened_remainder(df, sharp_dist_name, sharp_theta, sharp_measure)
    else:
        print(f"[exp_2] step5: skipped, theta={sharp_theta} not in theta grid {theta_grid}")

    # --- Step 6: combined power-law plot --------------------------------------
    combined_theta = 0.10
    if combined_theta in theta_grid:
        step6_combined_powerlaw_plot(df, fits_df, combined_theta, dist_names)
    else:
        print(f"[exp_2] step6: skipped, theta={combined_theta} not in theta grid {theta_grid}")

    # --- Step 7: final summary -------------------------------------------------
    print("[exp_2] step1 power-law fits (beta, expected ~0.5):")
    print(fits_df.to_string(index=False))
    print("[exp_2] step4 Wasserstein power-law fits (beta, expected ~0.5):")
    print(wass_fit_df.to_string(index=False))
    print(
        f"[exp_2] lambda_theta argmax = {theta_lambda_star} (lambda={lambda_star}); "
        f"lambda_theta/theta argmax = {theta_ratio_argmax:.4f} (ratio={ratio_argmax:.4f})"
    )

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[exp_2] total runtime: {time.time() - t0:.1f}s")
    print(f"[exp_2] peak memory (ru_maxrss): {peak_kib} KiB = {peak_kib / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
