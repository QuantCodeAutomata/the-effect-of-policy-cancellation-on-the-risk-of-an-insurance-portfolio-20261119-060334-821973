# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of five numerical experiments from the paper *"The effect of
policy cancellation on the risk of an insurance portfolio"*:

- `exp_1` — constant cancellation regime: Edgeworth / Cornish-Fisher
  asymptotics for VaR and L^p-Lipschitz risk measures (Section 3,
  Theorem 3.3, Theorem 3.8).
- `exp_2` — proportional cancellation in a homogeneous portfolio: central
  limit theorem in Wasserstein distance for VaR and W_p-continuous risk
  measures (Section 4.1-4.2, Lemma 4.1, Theorem 4.3).
- `exp_3` — heterogeneous portfolio and stochastic (Bernoulli) cancellation
  (Section 4.3-4.4, Lemma 4.5, Example 4.7, Lemma 4.8).
- `exp_4` — cancellation cost for term life insurance via Hattendorff's
  theorem, standalone and aggregated portfolio (Section 5.2).
- `exp_5` — cancellation cost for pure endowment insurance via Hattendorff's
  theorem, standalone and aggregated portfolio (Section 5.3).

No market data is used: `exp_1`-`exp_3` use synthetic i.i.d. Monte Carlo
samples, and `exp_4`-`exp_5` are closed-form recursions on the Austrian
mortality table printed in the paper's Table 1. No API keys are required.

## Layout

```
src/            flat library modules: edgeworth.py (Edgeworth/Cornish-Fisher
                expansions, cumulants), risk_measures.py (empirical VaR/TVaR,
                Wasserstein distance), proportional.py (homogeneous
                proportional-cancellation asymptotics), heterogeneous.py
                (multi-segment and Bernoulli cancellation), actuarial.py
                (Hattendorff backward recursion, CCR formulas), config.py
                (all experiment parameters as module-level variables)
exp/            one script per experiment; each is run as python -m exp.exp_N
data/           generate_synthetic.py: i.i.d. loss sample generators used by
                exp_1-exp_3; no data_loader.py because no experiment uses
                market data
tests/          pytest unit tests for src/
results/        tables, figures and RESULTS.md with the paper comparison
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest                        # full suite
python -m pytest -m "not market_data"    # equivalent here: no test is marked market_data
```

Experiments (reduced, time-bounded configuration by default; the paper's full
configuration with `--full`):

```bash
python -m exp.exp_1            # reduced grid
python -m exp.exp_1 --full     # paper's full grid: n up to 1,000,000, 100,000 MC reps
python -m exp.exp_2
python -m exp.exp_2 --full
python -m exp.exp_3
python -m exp.exp_3 --full
python -m exp.exp_4             # closed-form, fixed paper parameters, no --full
python -m exp.exp_5
```

`./reproduce.sh` runs the test suite and all five experiments at their
default configuration.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables, grouped by experiment, each with a comment stating
  its meaning, unit and paper source (or "not specified in the paper"). Do
  not hardcode parameters elsewhere.
- Where a parameter was reduced to fit the available computation time, both
  the paper value (`*_PAPER`) and the executed value (`*_RUN`) are kept as
  adjacent variables in `src/config.py`, and the experiment's `--full` flag
  selects the paper values.
- Data preparation code lives only in `data/`: synthetic sample generation in
  `data/generate_synthetic.py`. `src/` and `exp/` never generate or load data
  themselves.
- Experiments run from the repository root as `python -m exp.<name>`; tests
  as `python -m pytest`. Neither modifies `sys.path`.
- Running an experiment script overwrites its output files in `results/` in
  place. The numbers and verdicts in `results/RESULTS.md` are written by
  hand from the files produced by the most recent run and must be
  re-checked and updated after any rerun; they are not regenerated
  automatically.
- No API keys are used by this repository. If a future experiment were to
  need market data, it would be loaded only through `data/data_loader.py`
  using the QuantX Dataloader and Massive REST APIs, with keys read from an
  uncommitted `.env` (see `.env.example` conventions elsewhere), never
  hardcoded or logged.
