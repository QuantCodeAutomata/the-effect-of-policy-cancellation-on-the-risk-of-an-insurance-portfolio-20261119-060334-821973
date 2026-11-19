"""exp_5: Numerical cancellation costs for pure endowment insurance.

Reproduces, for a 20-year pure endowment contract on a life aged 40
(Austrian mortality table, Section 5.3):

* Table 4 — prospective reserve V_k(a) and conditional variance
  Var(L_k | J_k = a) from the Hattendorff backward recursion (eq. 5.6);
* Table 5 — standalone cancellation cost CCR_hat_k (eq. 5.1) for a single
  new-business cohort of n = 50,000 contracts, for theta in {1%, 5%, 10%};
* Table 6 — aggregated cancellation cost CCR_hat_k^aggr (eq. 5.2) for a
  portfolio of 20 annual cohorts of 50,000 contracts each;
* Figure 2 — relative cancellation costs CCR_hat_k / V_k(a) and
  CCR_hat_k^aggr / V_k(a), standalone vs aggregated.

All computations use the shared Hattendorff recursion of
``src/actuarial.py`` and the paper parameters of ``src/config.py``; the
table-building and plotting logic mirrors ``exp/exp_4.py``, with the
pure-endowment recursion and premium replacing the term life ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.actuarial import (
    net_premium_pure_endowment,
    pure_endowment_recursion,
    standalone_ccr,
    build_aggregated_portfolio,
    aggregated_ccr,
)

RESULTS_DIR = __file__.rsplit("/exp/", 1)[0] + "/results"


def build_table_4(q: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Table 4: reserve and conditional variance, k = 0, ..., T."""
    premium = net_premium_pure_endowment(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT)
    v_k, var_k = pure_endowment_recursion(q, config.DISCOUNT_FACTOR, config.ENDOWMENT_BENEFIT, premium)
    table = pd.DataFrame({
        "k": np.arange(v_k.size),
        "V_k(a)": v_k,
        "Var(L_k|J_k=a)": var_k,
        "sigma_k": np.sqrt(var_k),
    })
    return table, premium


def build_table_5(var_k: np.ndarray) -> pd.DataFrame:
    """Table 5: standalone cancellation cost CCR_hat_k (eq. 5.1) for the
    new-business cohort, k = 1, ..., T-1, under each cancellation rate."""
    rows = []
    for theta in config.CANCELLATION_THETAS:
        for k in range(1, config.CONTRACT_TERM):
            sigma_k = np.sqrt(var_k[k])
            n_k = config.COHORT_SIZE * (1.0 - theta) ** (k - 1)
            ccr = standalone_ccr(sigma_k, n_k, theta, config.VAR_Z)
            rows.append({"theta": theta, "k": k, "n_k": n_k, "sigma_k": sigma_k, "CCR_hat_k": ccr})
    return pd.DataFrame(rows)


def build_table_6(var_k: np.ndarray) -> pd.DataFrame:
    """Table 6: aggregated cancellation cost CCR_hat_k^aggr (eq. 5.2) for
    the portfolio of ``N_AGGREGATED_COHORTS`` annual cohorts, k = 1, ..., T-1,
    using the same cohort weighting as the term life Table 3 (exp/exp_4.py)."""
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


def plot_relative_costs(table4: pd.DataFrame, table5: pd.DataFrame, table6: pd.DataFrame, path: str) -> None:
    """Figure 2: relative cancellation costs CCR_hat_k / V_k(a), standalone
    vs aggregated, for each cancellation rate theta."""
    fig, ax = plt.subplots(figsize=(8, 5))
    v_k = table4.set_index("k")["V_k(a)"]
    for theta in config.CANCELLATION_THETAS:
        t5 = table5[table5["theta"] == theta].set_index("k")
        t6 = table6[table6["theta"] == theta].set_index("k")
        rel_standalone = t5["CCR_hat_k"] / v_k.loc[t5.index]
        rel_aggr = t6["CCR_hat_k_aggr"] / v_k.loc[t6.index]
        ax.plot(t5.index, rel_standalone, marker="o", markersize=3, label=f"standalone, θ={theta:.0%}")
        ax.plot(t6.index, rel_aggr, marker="s", markersize=3, linestyle="--", label=f"aggregated, θ={theta:.0%}")
    ax.set_xlabel("k (policy year)")
    ax.set_ylabel("CCR_hat_k / V_k(a)")
    ax.set_yscale("log")
    ax.set_title("Pure endowment insurance: relative cancellation cost, standalone vs aggregated")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    q = np.array(config.MORTALITY_Q, dtype=np.float64)
    assert q.size == config.CONTRACT_TERM, "mortality table must cover the full contract term"

    table4, premium = build_table_4(q)
    var_k = table4["Var(L_k|J_k=a)"].to_numpy()
    print(f"[exp_5] net annual premium P = {premium:.2f} (paper: {config.PURE_ENDOWMENT_PREMIUM_PAPER})")

    table5 = build_table_5(var_k)
    table6 = build_table_6(var_k)

    table4.to_csv(f"{RESULTS_DIR}/exp5_table4_reserve_variance.csv", index=False)
    table5.to_csv(f"{RESULTS_DIR}/exp5_table5_standalone_ccr.csv", index=False)
    table6.to_csv(f"{RESULTS_DIR}/exp5_table6_aggregated_ccr.csv", index=False)
    plot_relative_costs(table4, table5, table6, f"{RESULTS_DIR}/exp5_figure2_relative_cost.png")

    print("[exp_5] Table 4 (k=0,1,10,19,20):")
    print(table4[table4["k"].isin([0, 1, 10, 19, 20])].to_string(index=False))
    print("[exp_5] Table 5 summary (min/max CCR_hat_k by theta):")
    print(table5.groupby("theta")["CCR_hat_k"].agg(["min", "max"]))
    print("[exp_5] Table 6 summary (min/max CCR_hat_k^aggr by theta):")
    print(table6.groupby("theta")["CCR_hat_k_aggr"].agg(["min", "max"]))


if __name__ == "__main__":
    main()
