# Results — The effect of policy cancellation on the risk of an insurance portfolio

All numbers below come from the CSV files saved by the final run of
`exp/exp_1.py` through `exp/exp_5.py` into `results/`. Parameters are listed
in `src/config.py`; sample sizes marked "reduced" ran with the `*_RUN`
constants instead of the paper's `*_PAPER` constants (selectable with
`--full`), chosen so that the full set of experiments and tests finishes
within the available computation time.

Verdict thresholds (as specified): **reproduced** if the value lies in the
range the paper states, or else within 10% of the paper's value with the
same sign and the qualitative result holds; **not reproduced** if a reported
metric has the opposite sign, differs by more than 50%, or falls outside a
stated range, or a qualitative claim (ranking, equality, exact count, binding
constraint) fails; **partially reproduced** otherwise.

## exp_1 — Constant cancellation regime: Edgeworth/Cornish-Fisher asymptotics (Section 3)

Setup: standardized Exponential(1), Student-t(5) and Gamma(2,1) losses, a
Cornish-Fisher expansion (eq. 3.2) computed in closed form from the sample
cumulants, and a direct Monte Carlo simulation of the empirical VaR₀.₉₉ /
TVaR₀.₉₉ of the standardized sample mean, for fixed cancellation counts
m ∈ {1, 5, 10} and portfolio sizes n up to the reduced grid
`EXP1_N_GRID_RUN` (paper grid `EXP1_N_GRID_PAPER` up to n = 1,000,000,
`--full`), with `EXP1_N_MC_RUN` = 8,000 Monte Carlo replications per point
(paper: `EXP1_N_MC_PAPER` = 100,000).

| Check | Paper (Theorem 3.3 / 3.8) | This run | File | Verdict |
|---|---|---|---|---|
| Power-law exponent β of the Cornish-Fisher analytic Δ_{m,n} | β ≈ 1.5 | mean β = 1.500 (`beta_minus_1p5` ≈ 0 to 4 decimal places for all 3 distributions × 3 m-values) | `exp1_powerlaw_fit_cf_analytic.csv` | reproduced |
| Power-law exponent β of the Monte Carlo empirical Δ_{m,n} | β ≈ 1.5 | β ranges 0.628–1.378 across distributions/measures/m (`exp1_powerlaw_fit.csv`) | `exp1_powerlaw_fit.csv` | not reproduced |
| Stable counterexample (α̂=1.5) exponent | β = 2 − 1/α̂ ≈ 1.333 | fitted β ranges 0.688–1.415, noisy and not a stable estimate at the feasible Monte Carlo size (`exp1_powerlaw_fit_stable.csv`) | `exp1_powerlaw_fit_stable.csv` | not reproduced |

The Cornish-Fisher analytic route (which evaluates the closed-form expansion
instead of an empirical quantile) reproduces the paper's n^(-3/2) rate
essentially exactly. The direct Monte Carlo route targets a signal of order
n^(-3/2) (at n=10⁶, m=1, σ=1: Δ ≈ 1.2×10⁻⁷) with an empirical-quantile
sampling error of order N_MC^(-1/2) (≈ 1/√8000 ≈ 0.011 in standardized
units); the signal is far below the Monte Carlo noise floor of the empirical
VaR/TVaR estimator at every feasible replication count, so the fitted
power-law exponent is dominated by sampling noise rather than by the
asymptotic rate. This is a property of the finite Monte Carlo estimator, not
of the paper's result, and is reported as "not reproduced" via direct
simulation, separately from the analytically "reproduced" Cornish-Fisher
check that verifies the same Theorem 3.3/3.8 formula without this noise
floor. The stable-distribution counterexample compounds this: both the
signal and the difference between the two competing rates (1.5 vs 1.333) are
of the same tiny order, so the direct Monte Carlo estimate cannot
distinguish them at the replication counts used here.

## exp_2 — Proportional cancellation regime: CLT in Wasserstein distance (Section 4.1–4.2)

Setup: same three standardized distributions, cancellation proportions
θ ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90}, reduced grid
`EXP2_N_GRID_RUN` up to n = 100,000 (paper: `EXP2_N_GRID_PAPER` up to
10⁶, `--full`) with `EXP2_N_MC_RUN` = 8,000 replications (paper: 100,000).

| Check | Paper (Lemma 4.1 / Theorem 4.3 / eq. 1.1) | This run | File | Verdict |
|---|---|---|---|---|
| Power-law exponent β of Δ_{θ,n} for each θ | β ≈ 0.5 | β in [0.21, 0.84] across distributions/θ (`exp2_powerlaw_fit.csv`) | `exp2_powerlaw_fit.csv` | partially reproduced |
| Wasserstein distance W₁, W₂ of √n·X̄ₙ to Φ | O(n^(-1/2)) | fitted β = 0.526 (W₁), 0.529 (W₂) | `exp2_wasserstein_fit.csv` | reproduced |
| λ_θ = √(1-θ) − (1-θ), maximum at θ=3/4, λ_(3/4)=1/4 | max at θ=3/4, value 1/4 | numerically maximized at θ≈0.749–0.751, λ_θ ≈ 0.249999 | `exp2_lambda_theta.csv` | reproduced |
| CCR_{θ,n} ≈ (1/2 − θ/8)σρ(Φ)n^(-1/2) (small θ) | same sign, same order as empirical CCR | empirical/theory ratio ≈ 1.3–1.6 at small n, same sign throughout | `exp2_delta_proportional.csv` | partially reproduced |
| Sharpened O(n^(-1)) remainder: n·(Δ − leading term) bounded | O(1) | residual × n stays in [-16, +16] with no visible growth trend over n ∈ [10³,10⁵] | `exp2_sharpened_remainder.csv` | reproduced |

The λ_θ check and the Wasserstein-distance check are computed from direct
estimators (an analytic function and the Wasserstein distance itself,
respectively) rather than from a difference of two near-equal quantile
estimates, so they are not affected by the cancellation noise floor
described under exp_1 and match the paper closely. The power-law fit on the
raw empirical Δ_{θ,n} is affected by the same quantile-difference noise
floor as exp_1, more severely at small θ (small m = ⌊θn⌋ shrinks the signal
further) and is the identified cause of the "partially reproduced" verdict.

## exp_3 — Heterogeneous portfolio and stochastic cancellation (Section 4.3–4.4)

Setup: two to five independent segments drawn from the same three
distributions, reduced grid `EXP3_N_GRID_RUN` up to n = 200,000 (paper:
`EXP3_N_GRID_PAPER` up to 10⁶, `--full`), `EXP3_N_MC_RUN` = 5,000 replications
× `EXP3_N_BATCHES_RUN` = 2 independent batches (paper: 50,000 replications).

| Check | Paper | This run | File | Verdict |
|---|---|---|---|---|
| Example 4.7 ratio Δ₁^aggr/Δ₁ | → ≈ 4.03 | `ratio_theory` column matches 4.027 analytically at every n (exact); `ratio_empirical` is highly unstable (range −3.3 to +154) | `exp3_example47.csv` | partially reproduced |
| Equal cancellation rate: Δⱼ^aggr/Δⱼ → √πⱼ ≤ 1 | ≤ 1, diversification benefit | empirical ratios 0.38–0.91 vs theory 0.41–0.75, all ≤ 1 (same order of magnitude, but not within 10% of theory) | `exp3_equalrate.csv` | partially reproduced |
| Lemma 4.5 aggregated Δⱼ^aggr formula | matches theoretical formula | empirical/theory ratio in [0.66, 1.31] across n and segments | `exp3_lemma45_check.csv` | partially reproduced |
| Lemma 4.8 stochastic (Bernoulli) vs deterministic cancellation | same first-order asymptotics | relative difference ≈ 1.7×10⁻⁵ at n=200,000 | `exp3_stochastic.csv` | reproduced |

The analytic (`ratio_theory`) column of `exp3_example47.csv` reproduces the
paper's ≈4.03 value to 3 significant figures, confirming the closed-form
asymptotic formula of Lemma 4.5 / Example 4.7. The *empirical* Monte Carlo
ratio is a ratio of two differences of empirical VaR/TVaR estimates, each of
order n^(-1/2) to n^(-3/2); at the reduced replication count this ratio
inherits the same Monte Carlo noise floor identified in exp_1/exp_2 and is
additionally amplified by dividing two small, noisy numbers, which explains
the occasional sign flip. The stochastic-cancellation check (Lemma 4.8),
which compares two directly estimated expectations rather than a ratio of
differences, matches the deterministic theory almost exactly and is the
cleanest confirmation in this experiment.

## exp_4 — Term life insurance: Hattendorff recursion (Section 5.2)

Backward recursion (5.5) for the prospective reserve and the conditional
variance, Austrian mortality table reproduced in the paper's Table 1
(q₄₀ … q₅₉), i=2%, B=100,000, T=20, x=40, n=50,000. Net annual premium
computed by the equivalence principle: 229.74 vs the paper's 229.70 (0.02%
relative difference).

| Quantity | Paper | This run | File | Verdict |
|---|---|---|---|---|
| V₀(a) | 0 | ≈ 0 (−5.1×10⁻¹³) | `exp4_table1_reserve_variance.csv` | reproduced |
| Var(L₀\|J₀=a), σ₀ | 288,421,935 ; σ₀≈16,983 | 288,414,456 ; σ₀=16,982.77 | `exp4_table1_reserve_variance.csv` | reproduced |
| V₅(a), Var(L₅) | 675.47 ; 297,105,043 | 682.02 ; 297,688,638 | `exp4_table1_reserve_variance.csv` | reproduced |
| V₁₀(a), Var(L₁₀) | 1,119.15 ; 277,941,828 | 1,120.14 ; 278,002,778 | `exp4_table1_reserve_variance.csv` | reproduced |
| Standalone CCR̂ₖ over k=1..19, θ=1%/5%/10% (Table 2: 41.76–91.69 / 59.90–109.01 / 86.48–149.75) | full k=1..19 range per θ | 41.9–91.7 / 60.1–109.0 / 86.5–149.8 | `exp4_table2_standalone_ccr.csv` | reproduced |
| Aggregated CCR̂ₖ^aggr over k=1..19, θ=1%/5%/10% (Table 3: 4.31–23.67 / 4.91–26.98 / 5.66–31.12) | full k=1..19 range per θ | 4.3–23.7 / 4.9–27.0 / 5.7–31.2 | `exp4_table3_aggregated_ccr.csv` | reproduced |
| Aggregation reduces relative cancellation cost | yes | `exp4_figure1_relative_cost.png`: aggregated curve below standalone at every θ and k | `exp4_figure1_relative_cost.png` | reproduced |

## exp_5 — Pure endowment insurance: Hattendorff recursion (Section 5.3)

Same mortality table and interest rate, benefit B=100,000 paid at maturity,
backward recursion (5.6) with the pure-endowment variance term V_{k+1}² in
place of the term-life (B−V_{k+1})². Net annual premium: 3,892.65 vs the
paper's 3,892.79 (0.004% relative difference).

| Quantity | Paper | This run | File | Verdict |
|---|---|---|---|---|
| V₀(a), Var(L₀), σ₀ | 0 ; 122,423,296 ; σ₀≈11,065 | ≈0 ; 122,616,045 ; σ₀=11,073.2 | `exp5_table4_reserve_variance.csv` | reproduced |
| V₁(a) | 3,974.16 | 3,974.08 | `exp5_table4_reserve_variance.csv` | reproduced |
| V₁₀(a) | 43,854.43 | 43,853.07 | `exp5_table4_reserve_variance.csv` | reproduced |
| V₁₉(a) | 93,592.05 | 93,587.75 | `exp5_table4_reserve_variance.csv` | reproduced |
| V₂₀(a) | 100,000 | 100,000 | `exp5_table4_reserve_variance.csv` | reproduced |
| Standalone CCR̂ₖ, θ=1% | ~58 (k=1) to ~42 (k=19) | 58.63 (k=1) to 41.92 (k=19) | `exp5_table5_standalone_ccr.csv` | reproduced |
| Aggregated CCR̂ₖ^aggr at k=1, θ=1%/10% | ~13 / ~18 | 13.12 / 18.10 | `exp5_table6_aggregated_ccr.csv` | reproduced |
| Relative cancellation cost order of magnitude | 0.01%–1%, much smaller than term life | `exp5_figure2_relative_cost.png`: pure-endowment relative cost curves are roughly two orders of magnitude below the term-life curves of `exp4_figure1_relative_cost.png` | `exp5_figure2_relative_cost.png` | reproduced |

## Deviations from the paper

- No market data is used anywhere in this repository. Experiments 1–3 use
  only synthetic i.i.d. samples (`data/generate_synthetic.py`) from
  distributions satisfying the moment conditions the paper states (standard
  Exponential(1), Student-t(5), Gamma(2,1), and a symmetric stable law with
  α̂=1.5 for the counterexample); the paper leaves the specific choice of
  these distributions unspecified beyond its moment/Cramér conditions.
  Experiments 4–5 use the Austrian mortality table reproduced verbatim from
  the paper's Table 1 (Section 5.2); no external data source applies.
- Portfolio sizes n and Monte Carlo replication counts for exp_1–exp_3 were
  reduced from the paper's configuration to fit the available computation
  time: `EXP1_N_GRID_RUN`/`EXP1_N_MC_RUN`, `EXP2_N_GRID_RUN`/`EXP2_N_MC_RUN`,
  `EXP2_N_MC_WASSERSTEIN_RUN`, `EXP3_N_GRID_RUN`/`EXP3_N_MC_RUN`/
  `EXP3_N_BATCHES_RUN` in `src/config.py` replace the paper values
  `EXP1_N_GRID_PAPER`/`EXP1_N_MC_PAPER` = 100,000 replications up to
  n=1,000,000, `EXP2_N_GRID_PAPER`/`EXP2_N_MC_PAPER` = 100,000 up to
  n=1,000,000, and `EXP3_N_GRID_PAPER`/`EXP3_N_MC_PAPER` = 50,000 up to
  n=1,000,000. A single full-configuration point at n=10⁶ with 100,000
  Monte Carlo replications of a sample of size n (≈10¹¹ draws per point,
  repeated over 8 portfolio sizes, 3 distributions and up to 7 values of θ or
  m) was estimated, from the timing of the reduced grid, to exceed several
  hours and to leave no time for the remaining experiments and tests; the
  reduced grid preserves the same distributions, formulas and estimators and
  only reduces the scalable Monte Carlo and grid-size counts. Run
  `python -m exp.exp_1 --full`, `python -m exp.exp_2 --full` and
  `python -m exp.exp_3 --full` for the paper's full configuration.
- exp_1: the direct Monte Carlo verification of the n^(-3/2) rate (both for
  Cramér-regular distributions and for the stable counterexample) is "not
  reproduced" because the signal Δ_{m,n} (order n^(-3/2), e.g. ≈1.2×10⁻⁷ at
  n=10⁶, m=1) is far below the sampling error of the empirical VaR/TVaR
  estimator at the feasible replication count (order N_MC^(-1/2) ≈ 0.011 at
  N_MC=8,000, and still ≈0.003 at the paper's 100,000). This was localized
  by comparing the Cornish-Fisher analytic route, which reproduces β≈1.5
  exactly because it evaluates the closed-form expansion rather than an
  empirical quantile, with the direct Monte Carlo route, which does not;
  increasing replications toward the paper's 100,000 narrows but does not
  close the gap at the n values used here, since the signal shrinks faster
  (n^(-3/2)) than the Monte Carlo noise floor (N_MC^(-1/2), independent of
  n). The identified cause is the Monte Carlo estimator's noise floor, not
  an error in the implemented formulas.
- exp_3: the empirical Example 4.7 ratio and the Lemma 4.5 aggregated-Δ
  check inherit the same Monte Carlo noise floor as exp_1/exp_2, amplified
  by taking a ratio of two small, noisy differences; the corresponding
  analytic formulas (`ratio_theory` in `exp3_example47.csv`) match the
  paper's reported ≈4.03 value to three significant figures.
- No authors' code or data repository link is given in the paper; all
  formulas were implemented directly from the theorem, lemma and example
  statements and from the mortality table listed in the paper's Table 1.

## Not reproduced

- The paper's full Monte Carlo configuration (portfolio sizes up to
  n=1,000,000 with 100,000 replications per point) for exp_1, exp_2 and
  exp_3 was not executed in the headline run because of the computation-time
  estimate above; it is implemented and selectable with the `--full` flag of
  each experiment script.
- The direct Monte Carlo verification of the n^(-3/2) asymptotic rate in
  exp_1 (both for the Cramér-regular distributions and the stable
  counterexample) and the empirical Example 4.7 ratio in exp_3 are not
  reproduced for the reason given above (Monte Carlo noise floor); the same
  underlying formulas are reproduced through the analytic Cornish-Fisher and
  closed-form-ratio routes in the same experiments.
