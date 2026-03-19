"""
cards.py — Card representation and infinite-shoe sampler.

Under an infinite shoe every card draw is i.i.d.:
  ranks 2-9  → probability 1/13 each
  rank 10    → probability 4/13 (T, J, Q, K all share value 10)
  rank Ace   → probability 1/13  (value 11, reduced to 1 when needed)
"""
from __future__ import annotations
import numpy as np

# ── Card values ──────────────────────────────────────────────────────────────

# All distinct integer values a card can contribute
CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   # 11 = Ace

# Raw counts in a standard deck: ranks 2-9 (×4), rank-10 (×16), Ace (×4)
_RAW_COUNTS = {2:4, 3:4, 4:4, 5:4, 6:4, 7:4, 8:4, 9:4, 10:16, 11:4}
_TOTAL = sum(_RAW_COUNTS.values())   # 52

# Probability of drawing each card value from an infinite shoe
DRAW_PROBS: dict[int, float] = {v: c / _TOTAL for v, c in _RAW_COUNTS.items()}

# Numpy arrays for fast sampling
_VALUES_ARR  = np.array(CARD_VALUES, dtype=np.int32)          # shape (10,)
_PROBS_ARR   = np.array([DRAW_PROBS[v] for v in CARD_VALUES]) # shape (10,)

# Conditional probabilities for dealer peek:
#   given the hole card is NOT a 10-value  (upcard is Ace)
_NON10_MASK   = _VALUES_ARR != 10
_P_NON10      = _PROBS_ARR * _NON10_MASK
_P_NON10      = _P_NON10 / _P_NON10.sum()

#   given the hole card is NOT an Ace  (upcard is 10-value)
_NON_ACE_MASK = _VALUES_ARR != 11
_P_NON_ACE    = _PROBS_ARR * _NON_ACE_MASK
_P_NON_ACE    = _P_NON_ACE / _P_NON_ACE.sum()


def draw_card_value(rng: np.random.Generator) -> int:
    """Sample a single card value from the infinite shoe."""
    return int(rng.choice(_VALUES_ARR, p=_PROBS_ARR))


def draw_card_value_non10(rng: np.random.Generator) -> int:
    """Sample conditioned on NOT being a 10-value card (for dealer peek: Ace up)."""
    return int(rng.choice(_VALUES_ARR, p=_P_NON10))


def draw_card_value_non_ace(rng: np.random.Generator) -> int:
    """Sample conditioned on NOT being an Ace (for dealer peek: 10-value up)."""
    return int(rng.choice(_VALUES_ARR, p=_P_NON_ACE))


def draw_multiple(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n card values at once (faster than n individual calls)."""
    return rng.choice(_VALUES_ARR, size=n, p=_PROBS_ARR).astype(np.int32)


# ── Probability helpers used by oracle DP ───────────────────────────────────

def prob_of_value(value: int) -> float:
    """P(draw == value) under infinite shoe."""
    return DRAW_PROBS.get(value, 0.0)


def prob_of_value_non10(value: int) -> float:
    """P(draw == value | draw != 10) under infinite shoe."""
    if value == 10:
        return 0.0
    idx = CARD_VALUES.index(value)
    return float(_P_NON10[idx])


def prob_of_value_non_ace(value: int) -> float:
    """P(draw == value | draw != Ace) under infinite shoe."""
    if value == 11:
        return 0.0
    idx = CARD_VALUES.index(value)
    return float(_P_NON_ACE[idx])
