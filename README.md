# Blackjack Strategy Optimization Study

## Contents

- `blackjack_opt/`: simulator, oracle solver, optimizers, evaluation, and export code
- `main.tex`: main manuscript source
- `supplementary.tex`: standalone supplementary information
- `pyproject.toml`, `setup.py`: Python package metadata

## Requirements

- Python 3.11+
- `numpy`
- `pandas`
- `matplotlib`

## Installation

```bash
pip install -e .
```

## Reproducing core outputs

```bash
python -m blackjack_opt.run_experiment --figures-dir figures --results-dir results
```

This command generates the main result tables and figures used by the manuscript.
