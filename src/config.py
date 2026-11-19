"""Parameters for all experiments.

Paper: "The effect of policy cancellation on the risk of an insurance
portfolio". Every module-level variable states its meaning, unit and the
paper section it comes from; variables without a paper reference state that
the paper does not specify a value.

No market data is used anywhere in this repository: every experiment in the
paper relies on synthetic Monte Carlo samples (Section 3, 4) or on a single
published mortality table (Section 5). Accordingly ``data/data_loader.py`` is
not needed; only ``data/generate_synthetic.py`` is provided.
"""

from scipy.stats import norm

# ---------------------------------------------------------------------------
# Shared across experiments
# ---------------------------------------------------------------------------
ALPHA = 0.99  # VaR / TVaR confidence level used throughout (Section 5.2: "we use VaR_0.99")
RANDOM_SEED = 12345  # seed for the Monte Carlo generators; not specified in the paper

# ---------------------------------------------------------------------------
# exp_1: constant cancellation regime (Section 3, Theorem 3.3 / 3.8 / 3.12)
# ---------------------------------------------------------------------------
EXP1_DISTRIBUTIONS = ["exponential", "student_t5", "gamma21"]  # Cramer-condition distributions (data_requirements)
EXP1_STABLE_ALPHA_HAT = 1.5  # stability parameter of the counterexample (implementation_notes: alpha_hat in (0,2))
EXP1_N_GRID_PAPER = [1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]  # portfolio sizes n (data_requirements)
EXP1_N_GRID_RUN = [1000, 2000, 5000, 10000, 30000, 100000, 300000]  # reduced grid; see results/RESULTS.md
EXP1_M_VALUES = [1, 5, 10]  # fixed number of cancelled policies m (research_objectives)
EXP1_N_MC_PAPER = 100_000  # Monte Carlo replications (data_requirements: "at least 100,000")
EXP1_N_MC_RUN = 8_000  # reduced replication count; see results/RESULTS.md

# ---------------------------------------------------------------------------
# exp_2: proportional cancellation, homogeneous portfolio (Section 4.1-4.2)
# ---------------------------------------------------------------------------
EXP2_DISTRIBUTIONS = ["exponential", "gamma21", "student_t5"]  # data_requirements
EXP2_THETA_GRID = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]  # cancellation proportions (data_requirements)
EXP2_N_GRID_PAPER = [1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]  # portfolio sizes n (data_requirements)
EXP2_N_GRID_RUN = [1000, 2000, 5000, 10000, 30000, 100000]  # reduced grid; see results/RESULTS.md
EXP2_N_MC_PAPER = 100_000  # Monte Carlo replications (data_requirements)
EXP2_N_MC_RUN = 8_000  # reduced replication count; see results/RESULTS.md

# Lemma 4.2 Wasserstein-distance check uses its own Monte Carlo size and its
# own n-grid, decoupled from EXP2_N_MC_*/EXP2_N_GRID_*. The check estimates
# W_p(Law(sqrt(n)*X_bar_n), Phi) from an n_mc-sized Monte Carlo sample of
# sqrt(n)*X_bar_n matched against the exact quantile function of Phi; this
# estimator carries its own O(n_mc^{-1/2}) sampling-noise floor (the
# empirical-measure-to-population-law Wasserstein distance of a finite
# n_mc-sample, not specified in the paper) that swamps the O(n^{-1/2})
# CLT signal of Lemma 4.2 unless n_mc is kept much larger than every n in
# the grid used here.
EXP2_N_MC_WASSERSTEIN_PAPER = 2_000_000  # MC replications for the Wasserstein check (not specified in the paper)
EXP2_N_GRID_WASSERSTEIN_PAPER = [1000, 2000, 5000, 10000, 20000, 50000]  # kept << EXP2_N_MC_WASSERSTEIN_PAPER so the n_mc^{-1/2} floor stays negligible
EXP2_N_MC_WASSERSTEIN_RUN = 300_000  # reduced MC size for the Wasserstein check; see results/RESULTS.md
EXP2_N_GRID_WASSERSTEIN_RUN = [500, 1000, 2000, 5000, 10000]  # reduced n-grid, kept << EXP2_N_MC_WASSERSTEIN_RUN

# ---------------------------------------------------------------------------
# exp_3: heterogeneous portfolio and stochastic cancellation (Section 4.3-4.4)
# ---------------------------------------------------------------------------
# Example 4.7 (Section 4.3): two segments, equal weight, equal sigma, different theta
EXP3_EX47_WEIGHTS = [0.5, 0.5]  # w_1 = w_2 = 0.5 (Example 4.7)
EXP3_EX47_SIGMAS = [1.0, 1.0]  # sigma_1 = sigma_2 (Example 4.7, standardized losses)
EXP3_EX47_THETAS = [0.01, 0.10]  # theta_1 = 1%, theta_2 = 10% (Example 4.7)
EXP3_EX47_RATIO_EXPECTED = 4.03  # Delta_1^aggr / Delta_1 -> ~4.03 (research_objectives, Example 4.7)
EXP3_N_GRID_PAPER = [10_000, 50_000, 100_000, 500_000, 1_000_000]  # portfolio sizes n (data_requirements)
EXP3_N_GRID_RUN = [10_000, 50_000, 200_000]  # reduced grid; see results/RESULTS.md
EXP3_N_MC_PAPER = 50_000  # Monte Carlo replications for the stochastic case (data_requirements)
EXP3_N_MC_RUN = 5_000  # reduced replication count; see results/RESULTS.md
EXP3_N_BATCHES_PAPER = 20  # independent batches averaged per point, >=50_000*20 total MC draws (data_requirements)
EXP3_N_BATCHES_RUN = 2  # reduced-run batch count; averaging reduces VaR_0.99 quantile-estimation noise
# Equal-rate consistency check (verifies Delta_j^aggr/Delta_j -> sqrt(pi_j))
EXP3_EQUALRATE_WEIGHTS = [0.2, 0.3, 0.5]  # l=3 segments, not specified in the paper beyond moment conditions
EXP3_EQUALRATE_SIGMAS = [1.0, 1.5, 0.8]  # segment standard deviations, not specified in the paper
EXP3_EQUALRATE_THETA = 0.05  # common cancellation rate theta_j = theta for all j (research_objectives)

# ---------------------------------------------------------------------------
# exp_4 / exp_5: Hattendorff recursion for term life and pure endowment (Section 5)
# ---------------------------------------------------------------------------
# Austrian mortality table, ages 40-59, one-year death probabilities q_x
# (Statistik Austria, Life table for Austria 2020-2022, males and females
# combined [SA23], as reproduced in Table 1 of the paper, Section 5.2).
MORTALITY_AGES = list(range(40, 60))
MORTALITY_Q = [
    0.0009, 0.0010, 0.0010, 0.0011, 0.0012, 0.0014, 0.0015, 0.0016, 0.0018, 0.0020,
    0.0022, 0.0024, 0.0027, 0.0030, 0.0033, 0.0037, 0.0041, 0.0046, 0.0051, 0.0057,
]
MORTALITY_TABLE = dict(zip(MORTALITY_AGES, MORTALITY_Q))

ISSUE_AGE = 40  # insured age x at inception (Section 5.2)
CONTRACT_TERM = 20  # maturity T in years (Section 5.2)
DEATH_BENEFIT = 100_000.0  # death benefit B for term life (Section 5.2)
ENDOWMENT_BENEFIT = 100_000.0  # survival benefit B for pure endowment (Section 5.3)
INTEREST_RATE = 0.02  # annual effective interest rate i (Section 5.2)
DISCOUNT_FACTOR = 1.0 / (1.0 + INTEREST_RATE)  # v = 1/(1+i) (Section 5.2)

TERM_LIFE_PREMIUM_PAPER = 229.70  # net annual premium reported in the paper (Section 5.2)
PURE_ENDOWMENT_PREMIUM_PAPER = 3_892.79  # net annual premium reported in the paper (Section 5.3)

COHORT_SIZE = 50_000  # new-business cohort size n (Section 5.2 / 5.3)
CANCELLATION_THETAS = [0.01, 0.05, 0.10]  # constant annual cancellation probabilities theta (Section 5.2)
N_AGGREGATED_COHORTS = 20  # number of annual cohorts forming the aggregated portfolio (Section 5.2)

VAR_Z = float(norm.ppf(ALPHA))  # Phi^{-1}(0.99) approx 2.3263 (Section 5.2)
