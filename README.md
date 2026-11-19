# The effect of policy cancellation on the risk of an insurance portfolio

Reproduction of selected numerical experiments from the paper *"The effect of
policy cancellation on the risk of an insurance portfolio"*.

This repository implements five experiments from the paper:

1. **exp_1** — Constant cancellation regime: Edgeworth / Cornish-Fisher
   asymptotics for VaR and L^p-Lipschitz risk measures (Section 3,
   Theorem 3.3, Theorem 3.8).
2. **exp_2** — Proportional cancellation regime in a homogeneous portfolio:
   central limit theorem in Wasserstein distance for VaR and W_p-continuous
   risk measures (Section 4.1-4.2, Lemma 4.1, Theorem 4.3).
3. **exp_3** — Heterogeneous portfolio and stochastic cancellation: multi
   segment proportional cancellation with different rates, and Bernoulli
   cancellation (Section 4.3-4.4, Lemma 4.5, Example 4.7, Lemma 4.8).
4. **exp_4** — Numerical cancellation costs for term life insurance via
   Hattendorff's theorem, standalone and aggregated portfolio (Section 5.2).
5. **exp_5** — Numerical cancellation costs for pure endowment insurance via
   Hattendorff's theorem, standalone and aggregated portfolio (Section 5.3).

No market data is used: experiments 1-3 rely on synthetic Monte Carlo samples
drawn from distributions satisfying the paper's moment conditions, and
experiments 4-5 are closed-form actuarial recursions on the published
Austrian mortality table reproduced in the paper's Table 1 (Section 5.2).

## Repository layout

```
src/    library modules (Edgeworth/Cornish-Fisher, proportional and
        heterogeneous cancellation asymptotics, risk measures, Hattendorff
        recursion, configuration)
exp/    one script per experiment, run with `python -m exp.exp_N`
data/   data/generate_synthetic.py: synthetic i.i.d. loss sample generation
tests/  pytest unit tests for src/
results/ tables, figures and results/RESULTS.md with the paper comparison
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys are required: this paper's experiments use only synthetic data
and a published mortality table.

## Tests

```bash
python -m pytest
```

## Running the experiments

Each experiment can be run at a reduced configuration (default, chosen to
finish within the available computation time; see `results/RESULTS.md`) or
at the paper's full configuration with `--full`:

```bash
python -m exp.exp_1            # reduced grid
python -m exp.exp_1 --full     # paper's full grid (n up to 1,000,000, 100,000 MC reps)
python -m exp.exp_2
python -m exp.exp_2 --full
python -m exp.exp_3
python -m exp.exp_3 --full
python -m exp.exp_4
python -m exp.exp_5
```

Experiments 4 and 5 are closed-form recursions on fixed paper parameters and
have no reduced/full distinction; they run in well under a second.

Running an experiment script overwrites its output files in `results/`. The
numbers and verdicts in `results/RESULTS.md` are written by hand from the
files produced by the most recent run and must be re-checked after a rerun.

## reproduce.sh

```bash
./reproduce.sh
```

runs the test suite and every experiment at its default (time-bounded)
configuration, writing all outputs to `results/`.
