"""exp_4: Numerical cancellation costs for term life insurance.

Reproduces, for a 20-year term life contract on a life aged 40 (Austrian
mortality table, Section 5.2):

* Table 1 — prospective reserve V_k(a) and conditional variance
  Var(L_k | J_k = a) from the Hattendorff backward recursion (eq. 5.5);
* Table 2 — standalone cancellation cost CCR_hat_k (eq. 5.1) for a single
  new-business cohort of n = 50,000 contracts, for theta in {1%, 5%, 10%};
* Table 3 — aggregated cancellation cost CCR_hat_k^aggr (eq. 5.2) for a
  portfolio of 20 annual cohorts of 50,000 contracts each;
* Figure 1 — relative cancellation costs CCR_hat_k / V_k(a) and
  CCR_hat_k^aggr / V_k(a), standalone vs aggregated.

All computations use the shared Hattendorff recursion of
``src/actuarial.py`` and the paper parameters of ``src/config.py``; no
private shortcuts are defined here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.actuarial import (
    net_premium_term_life,
    term_life_recursion,
    standalone_ccr,
    build_aggregated_portfolio,
    aggregated_ccr,
)

RESULTS_DIR = __file__.rsplit("/exp/", 1)[0] + "/results"


def build_table_1(q: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Table 1: reserve and conditional variance, k = 0, ..., T."""
    premium = net_premium_term_life(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT)
    v_k, var_k = term_life_recursion(q, config.DISCOUNT_FACTOR, config.DEATH_BENEFIT, premium)
    table = pd.DataFrame({
        "k": np.arange(v_k.size),
        "V_k(a)": v_k,
        "Var(L_k|J_k=a)": var_k,
        "sigma_k": np.sqrt(var_k),
    })
    return table, premium


def build_table_2(var_k: np.ndarray) -> pd.DataFrame:
    """Table 2: standalone cancellation cost CCR_hat_k (eq. 5.1) for the
    new-business cohort, k = 1, ..., T-1, under each cancellation rate."""
    rows = []
    for theta in config.CANCELLATION_THETAS:
        for k in range(1, config.CONTRACT_TERM):
            sigma_k = np.sqrt(var_k[k])
            n_k = config.COHORT_SIZE * (1.0 - theta) ** (k - 1)
            ccr = standalone_ccr(sigma_k, n_k, theta, config.VAR_Z)
            rows.append({"theta": theta, "k": k, "n_k": n_k, "sigma_k": sigma_k, "CCR_hat_k": ccr})
    return pd.DataFrame(rows)


def build_table_3(var_k: np.ndarray) -> pd.DataFrame:
    """Table 3: aggregated cancellation cost CCR_hat_k^aggr (eq. 5.2) for
    the portfolio of ``N_AGGREGATED_COHORTS`` annual cohorts, k = 1, ..., T-1.

    Cohort aged k (k years since issuance, in force for k-1 full
    cancellation years) has size n*(1-theta)^(k-1) and conditional variance
    Var(L_k|J_k=a); the aggregated portfolio pools cohorts aged 1,...,T-1
    (the age-0 cohort has just been issued and the age-T cohort has matured,
    so neither carries future cancellation risk relative to the others)."""
    rows = []
    for theta in config.CANCELLATION_THETAS:
        weights, sigmas2 = build_aggregated_portfolio(
            config.COHORT_SIZE, theta, config.N_AGGREGATED_COHORTS, var_k
        )
        total_n = config.COHORT_SIZE * sum(
            (1.0 - theta) ** (k - 1) for k in range(1, config.N_AGGREGATED_COHORTS)
        )
        thetas_vec = np.full_like(weights, theta)
        for idx, k in enumerate(range(1, config.N_AGGREGATED_COHORTS)):
            ccr = aggregated_ccr(
                sigmas2[idx], theta, weights, sigmas2, thetas_vec, total_n, config.VAR_Z
            )
            rows.append({"theta": theta, "k": k, "CCR_hat_k_aggr": ccr})
    return pd.DataFrame(rows)


def plot_relative_costs(table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame, path: str) -> None:
    """Figure 1: relative cancellation costs CCR_hat_k / V_k(a), standalone
    vs aggregated, for each cancellation rate theta."""
    fig, ax = plt.subplots(figsize=(8, 5))
    v_k = table1.set_index("k")["V_k(a)"]
    for theta in config.CANCELLATION_THETAS:
        t2 = table2[table2["theta"] == theta].set_index("k")
        t3 = table3[table3["theta"] == theta].set_index("k")
        rel_standalone = t2["CCR_hat_k"] / v_k.loc[t2.index]
        rel_aggr = t3["CCR_hat_k_aggr"] / v_k.loc[t3.index]
        ax.plot(t2.index, rel_standalone, marker="o", markersize=3, label=f"standalone, θ={theta:.0%}")
        ax.plot(t3.index, rel_aggr, marker="s", markersize=3, linestyle="--", label=f"aggregated, θ={theta:.0%}")
    ax.set_xlabel("k (policy year)")
    ax.set_ylabel("CCR_hat_k / V_k(a)")
    ax.set_yscale("log")
    ax.set_title("Term life insurance: relative cancellation cost, standalone vs aggregated")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    q = np.array(config.MORTALITY_Q, dtype=np.float64)
    assert q.size == config.CONTRACT_TERM, "mortality table must cover the full contract term"

    table1, premium = build_table_1(q)
    var_k = table1["Var(L_k|J_k=a)"].to_numpy()
    print(f"[exp_4] net annual premium P = {premium:.2f} (paper: {config.TERM_LIFE_PREMIUM_PAPER})")

    table2 = build_table_2(var_k)
    table3 = build_table_3(var_k)

    table1.to_csv(f"{RESULTS_DIR}/exp4_table1_reserve_variance.csv", index=False)
    table2.to_csv(f"{RESULTS_DIR}/exp4_table2_standalone_ccr.csv", index=False)
    table3.to_csv(f"{RESULTS_DIR}/exp4_table3_aggregated_ccr.csv", index=False)
    plot_relative_costs(table1, table2, table3, f"{RESULTS_DIR}/exp4_figure1_relative_cost.png")

    print("[exp_4] Table 1 (k=0,5,10,19,20):")
    print(table1[table1["k"].isin([0, 5, 10, 19, 20])].to_string(index=False))
    print("[exp_4] Table 2 summary (min/max CCR_hat_k by theta):")
    print(table2.groupby("theta")["CCR_hat_k"].agg(["min", "max"]))
    print("[exp_4] Table 3 summary (min/max CCR_hat_k^aggr by theta):")
    print(table3.groupby("theta")["CCR_hat_k_aggr"].agg(["min", "max"]))


if __name__ == "__main__":
    main()
