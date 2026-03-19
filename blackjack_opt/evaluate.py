"""
evaluate.py — Evaluation and metric computation.

Two modes:
  A. Unconditional EV: simulate full rounds from natural deal distribution.
  B. Cell-conditional regret: compare learned policy vs oracle per cell.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from blackjack_opt.config import NUM_ACTIONS, ACTION_NAMES
from blackjack_opt.env import BlackjackEnv
from blackjack_opt.oracle import OracleResult
from blackjack_opt.policy import Policy
from blackjack_opt.state import DecisionCell
from blackjack_opt.rules import Rules


PolicyDict = dict[DecisionCell, int]


# ── Mode A: Unconditional EV ─────────────────────────────────────────────────

def evaluate_ev(
    env: BlackjackEnv,
    policy: PolicyDict | Policy,
    n_hands: int,
    rng: np.random.Generator,
) -> float:
    """
    Simulate n_hands under the given policy.
    Returns mean net reward per hand (per unit bet).
    """
    total = 0.0
    for _ in range(n_hands):
        state, cell = env.reset(bet=1.0)
        while not state.done:
            if cell is None:
                break
            if isinstance(policy, Policy):
                action = policy.get_action_with_rng(cell, rng)
            else:
                action = policy.get(cell, 0)   # default: stand
            state, cell, reward, done = env.step(state, action)
        total += state.reward
    return total / n_hands


# ── Mode B: Cell-conditional regret ─────────────────────────────────────────

def compute_regret_table(
    eval_policy: PolicyDict,
    oracle_result: OracleResult,
    rules: Rules,
) -> pd.DataFrame:
    """
    For each cell in the oracle, compare eval_policy vs optimal action.

    Returns DataFrame with columns:
      cell_str, player_total, dealer_upcard, is_soft, is_pair, pair_rank,
      split_depth, optimal_action, optimal_action_name,
      taken_action, taken_action_name, action_match, regret,
      q_optimal, q_taken
    """
    rows = []
    seen_cells: set[DecisionCell] = set()
    for cell, q_vec in oracle_result.Q_star.items():
        if cell in seen_cells:
            continue
        seen_cells.add(cell)

        optimal_action = oracle_result.pi_star[cell]
        q_opt = oracle_result.V_star[cell]

        taken_action = eval_policy.get(cell, 0)
        q_taken = q_vec[taken_action] if np.isfinite(q_vec[taken_action]) else q_opt
        regret   = q_opt - q_taken

        rows.append({
            "cell_str":          str(cell),
            "player_total":      cell.player_total,
            "dealer_upcard":     cell.dealer_upcard,
            "is_soft":           cell.is_soft,
            "is_pair":           cell.is_pair,
            "pair_rank":         cell.pair_rank,
            "split_depth":       cell.split_depth,
            "can_double":        cell.can_double,
            "can_split":         cell.can_split,
            "from_split":        cell.from_split,
            "optimal_action":    optimal_action,
            "optimal_name":      ACTION_NAMES[optimal_action],
            "taken_action":      taken_action,
            "taken_name":        ACTION_NAMES[taken_action],
            "action_match":      (optimal_action == taken_action),
            "regret":            regret,
            "q_optimal":         q_opt,
            "q_taken":           q_taken,
        })
    return pd.DataFrame(rows)


def action_match_rate(
    eval_policy: PolicyDict,
    oracle_result: OracleResult,
    weight_by_frequency: bool = False,
    cell_frequencies: dict[DecisionCell, float] | None = None,
) -> float:
    """Fraction of cells where eval_policy matches oracle_policy."""
    cells = list(oracle_result.pi_star.keys())
    if not cells:
        return 0.0

    matches = np.array([
        eval_policy.get(c, 0) == oracle_result.pi_star[c]
        for c in cells
    ], dtype=float)

    if weight_by_frequency and cell_frequencies:
        weights = np.array([cell_frequencies.get(c, 0.0) for c in cells])
        total_w = weights.sum()
        if total_w > 0:
            return float(np.dot(matches, weights) / total_w)

    return float(matches.mean())


def worst_cell_regret(regret_table: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    return (
        regret_table
        .sort_values("regret", ascending=False)
        .head(top_k)
        [["cell_str", "player_total", "dealer_upcard", "is_soft", "is_pair",
          "optimal_name", "taken_name", "regret", "q_optimal", "q_taken"]]
        .reset_index(drop=True)
    )


def compute_cell_visit_frequencies(
    env: BlackjackEnv,
    policy: PolicyDict | Policy,
    n_hands: int,
    rng: np.random.Generator,
) -> dict[DecisionCell, float]:
    """Relative frequency of visiting each decision cell."""
    counts: dict[DecisionCell, int] = {}
    total_decisions = 0

    for _ in range(n_hands):
        state, cell = env.reset(bet=1.0)
        while not state.done:
            if cell is None:
                break
            counts[cell] = counts.get(cell, 0) + 1
            total_decisions += 1
            if isinstance(policy, Policy):
                action = policy.get_action_with_rng(cell, rng)
            else:
                action = policy.get(cell, 0)
            state, cell, reward, done = env.step(state, action)

    if total_decisions == 0:
        return {}
    return {c: cnt / total_decisions for c, cnt in counts.items()}


def convergence_metrics(log_df: pd.DataFrame, oracle_ev: float) -> dict:
    """
    Compute convergence speed metrics from a training log DataFrame.
    log_df must have columns: hands, ev_per_hand.
    """
    if log_df.empty:
        return {"hands_to_95pct": None, "hands_to_99pct": None, "final_ev": None}

    first_ev = log_df["ev_per_hand"].iloc[0]
    gap      = oracle_ev - first_ev
    if abs(gap) < 1e-10:
        return {
            "hands_to_95pct": 0,
            "hands_to_99pct": 0,
            "final_ev":       float(log_df["ev_per_hand"].iloc[-1]),
        }

    result = {"final_ev": float(log_df["ev_per_hand"].iloc[-1])}
    for pct, key in [(0.95, "hands_to_95pct"), (0.99, "hands_to_99pct")]:
        threshold = first_ev + pct * gap
        row = log_df[log_df["ev_per_hand"] >= threshold]
        result[key] = int(row["hands"].iloc[0]) if not row.empty else None

    return result


def summarize_comparison(
    optimizer_logs: dict[str, pd.DataFrame],
    oracle_ev: float,
    oracle_policy: PolicyDict,
    eval_policies: dict[str, PolicyDict],
    oracle_result: OracleResult,
) -> pd.DataFrame:
    """
    One-row-per-optimizer summary table with all key metrics.
    """
    rows = []
    for name, log_df in optimizer_logs.items():
        conv = convergence_metrics(log_df, oracle_ev)
        epol = eval_policies.get(name, {})
        match = action_match_rate(epol, oracle_result) if epol else None
        regret_df = compute_regret_table(epol, oracle_result, None) if epol else None
        mean_regret = regret_df["regret"].mean() if regret_df is not None else None
        worst_regret = regret_df["regret"].max() if regret_df is not None else None

        total_hands = int(log_df["hands"].iloc[-1]) if not log_df.empty else 0
        final_ev    = conv["final_ev"]
        elapsed     = float(log_df["elapsed_s"].iloc[-1]) if "elapsed_s" in log_df.columns and not log_df.empty else None

        rows.append({
            "optimizer":       name,
            "total_hands":     total_hands,
            "final_ev":        round(final_ev, 5) if final_ev is not None else None,
            "oracle_ev":       round(oracle_ev, 5),
            "ev_gap":          round(oracle_ev - final_ev, 5) if final_ev is not None else None,
            "action_match":    round(match, 4) if match is not None else None,
            "mean_regret":     round(mean_regret, 5) if mean_regret is not None else None,
            "worst_regret":    round(worst_regret, 5) if worst_regret is not None else None,
            "hands_to_95pct":  conv.get("hands_to_95pct"),
            "hands_to_99pct":  conv.get("hands_to_99pct"),
            "elapsed_s":       round(elapsed, 1) if elapsed is not None else None,
        })
    return pd.DataFrame(rows)
