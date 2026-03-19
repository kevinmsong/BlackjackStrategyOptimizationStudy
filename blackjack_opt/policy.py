"""
policy.py — Logit-parameterized policy over abstract decision cells.

Provides differentiable (PG) and non-differentiable (SPSA, CEM) interfaces.
Masked softmax ensures illegal actions are never selected.
"""
from __future__ import annotations
import copy
import numpy as np

from blackjack_opt.config import NUM_ACTIONS, Rules
from blackjack_opt.state import DecisionCell


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Softmax over legal actions only.
    mask: bool array, True = legal action.
    Returns probability array (illegal actions get 0).
    """
    l = logits.copy().astype(np.float64)
    l[~mask] = -1e30
    l -= l[mask].max()          # numeric stability (only over legal actions)
    exp_l = np.exp(l)
    exp_l[~mask] = 0.0
    s = exp_l.sum()
    if s == 0:
        # Fallback: uniform over legal actions
        probs = mask.astype(np.float64)
        return probs / probs.sum()
    return exp_l / s


class Policy:
    """
    Tabular logit policy: theta[cell_idx, action] → probs via masked softmax.
    """

    def __init__(
        self,
        rules: Rules,
        cells: list[DecisionCell],
        init_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rules      = rules
        self.cells      = list(cells)
        self.n_cells    = len(cells)
        self.cell_to_idx: dict[DecisionCell, int] = {c: i for i, c in enumerate(cells)}

        # Build legal masks
        from blackjack_opt.oracle import _compute_q
        self.legal_masks = np.zeros((self.n_cells, NUM_ACTIONS), dtype=bool)
        for i, cell in enumerate(cells):
            q = _compute_q(cell, rules)
            self.legal_masks[i] = np.isfinite(q)

        # Initialize logits
        _rng = rng or np.random.default_rng(0)
        self.theta = _rng.standard_normal((self.n_cells, NUM_ACTIONS)) * init_std

    # ── Query interface ──────────────────────────────────────────────────────

    def get_probs(self, cell: DecisionCell) -> np.ndarray:
        """Masked softmax probabilities for cell."""
        idx = self.cell_to_idx.get(cell)
        if idx is None:
            raise KeyError(f"Cell not in policy: {cell}")
        return masked_softmax(self.theta[idx], self.legal_masks[idx])

    def get_action(self, cell: DecisionCell, deterministic: bool = False) -> int:
        """
        Sample or greedy-select an action.
        deterministic=True → argmax of logits (for evaluation / chart export).
        """
        idx = self.cell_to_idx.get(cell)
        if idx is None:
            # Fall back to stand for unknown cells
            return 0
        if deterministic:
            logits = self.theta[idx].copy()
            logits[~self.legal_masks[idx]] = -1e30
            return int(np.argmax(logits))
        probs = masked_softmax(self.theta[idx], self.legal_masks[idx])
        return int(np.random.default_rng().choice(NUM_ACTIONS, p=probs))

    def get_action_with_rng(
        self, cell: DecisionCell, rng: np.random.Generator, deterministic: bool = False
    ) -> int:
        idx = self.cell_to_idx.get(cell)
        if idx is None:
            return 0
        if deterministic:
            logits = self.theta[idx].copy()
            logits[~self.legal_masks[idx]] = -1e30
            return int(np.argmax(logits))
        probs = masked_softmax(self.theta[idx], self.legal_masks[idx])
        return int(rng.choice(NUM_ACTIONS, p=probs))

    def get_logprob(self, cell: DecisionCell, action: int) -> float:
        probs = self.get_probs(cell)
        p = probs[action]
        return float(np.log(p + 1e-30))

    def entropy(self, cell: DecisionCell) -> float:
        probs = self.get_probs(cell)
        return float(-np.sum(probs * np.log(probs + 1e-30)))

    def mean_entropy(self) -> float:
        total = 0.0
        for i in range(self.n_cells):
            probs = masked_softmax(self.theta[i], self.legal_masks[i])
            total -= np.sum(probs * np.log(probs + 1e-30))
        return total / self.n_cells

    # ── Serialization (for SPSA / CEM) ──────────────────────────────────────

    def to_array(self) -> np.ndarray:
        """Flat copy of theta (n_cells × NUM_ACTIONS)."""
        return self.theta.flatten().copy()

    def from_array(self, theta_flat: np.ndarray) -> "Policy":
        """Return new Policy with given flat theta values."""
        p = self.clone()
        p.theta = theta_flat.reshape(self.n_cells, NUM_ACTIONS).copy()
        return p

    def clone(self) -> "Policy":
        p = Policy.__new__(Policy)
        p.rules        = self.rules
        p.cells        = self.cells
        p.n_cells      = self.n_cells
        p.cell_to_idx  = self.cell_to_idx
        p.legal_masks  = self.legal_masks
        p.theta        = self.theta.copy()
        return p

    # ── Chart export ─────────────────────────────────────────────────────────

    def argmax_chart(self) -> dict[DecisionCell, int]:
        """Deterministic policy table: cell → argmax action."""
        chart: dict[DecisionCell, int] = {}
        for cell in self.cells:
            chart[cell] = self.get_action(cell, deterministic=True)
        return chart
