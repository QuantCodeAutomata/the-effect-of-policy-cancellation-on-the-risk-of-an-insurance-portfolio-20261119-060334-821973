#!/usr/bin/env bash
# Runs the test suite and every experiment at its default, time-bounded
# configuration, writing all outputs to results/. Pass no arguments.
set -uo pipefail

python -m pytest -q
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
