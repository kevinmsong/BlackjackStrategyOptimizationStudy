# Blackjack Strategy Optimization Study

Code and manuscript for **Policy Recovery and No-Count Bet-Sizing Optimality in Infinite-Shoe Blackjack**. The study uses blackjack as a controlled benchmark for learning from simulated play: an exact dynamic-programming "oracle" gives the optimal action for every decision, and three model-free optimizers are measured against it.

## What it does

- **Oracle:** exact dynamic programming over 4,600 canonical decision cells for a fixed Vegas-style ruleset (S17, 3:2 blackjack, dealer peek, double on any two cards, double after split, resplit to four hands). It gives ground-truth action values and an oracle EV per hand.
- **Learned policies:** three optimizers trained only from simulated hands, each compared with the oracle by action-match rate, EV, and regret:
  - masked REINFORCE (policy gradient)
  - SPSA (simultaneous perturbation stochastic approximation)
  - CEM (cross-entropy method)
- **Bet sizing:** a study of bet-sizing strategies when the player does not count cards.
- **Outputs:** regret tables, CSV results, and the figures used in the manuscript.

## Contents

- `blackjack_opt/`: simulator, oracle solver, optimizers, evaluation, betting study, and export code
- `blackjack_opt/tests/`: unit tests for the environment, hands, rules, and oracle
- `main.tex`: main manuscript source
- `supplementary.tex`: standalone supplementary information
- `pyproject.toml`, `setup.py`: Python package metadata

## Requirements

- Python 3.11+
- `numpy`, `pandas`, `matplotlib`
- Optional: `pytest` for tests (`.[dev]`), `numba` for speed (`.[fast]`)

## Installation

```bash
pip install -e .            # or: pip install -e ".[dev]" to include pytest
```

## Run

Full run (all stages, default budgets). This can take a long time:

```bash
python -m blackjack_opt.run_experiment --figures-dir figures --results-dir results
```

Quick sanity run at 10% of the default budget:

```bash
python -m blackjack_opt.run_experiment --quick --figures-dir figures --results-dir results
```

The orchestrator runs these stages in order: solve the oracle, check the oracle against the simulator, train PG, SPSA, and CEM, build the evaluation and regret tables, run the bet-sizing study, export figures to `figures/`, and print a summary table. CSV results go to `results/`.

Useful options:

| Option | Default | Effect |
|---|---|---|
| `--seed` | `42` | Random seed |
| `--rules` | `VEGAS_S17` | Ruleset variant |
| `--pg-hands` | `1000000` | Training hands for REINFORCE |
| `--spsa-iter` | `8000` | SPSA iterations |
| `--cem-gen` | `300` | CEM generations |
| `--eval-hands` | `100000` | Hands per policy evaluation |
| `--no-pg` / `--no-spsa` / `--no-cem` / `--no-betting` | off | Skip a stage |
| `--variants` | off | Also run EV sensitivity across all rule variants |

## Tests

```bash
pip install -e ".[dev]"
pytest
```
