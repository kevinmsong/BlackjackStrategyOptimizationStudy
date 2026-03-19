"""
state.py — Abstract decision cell and full round state.

DecisionCell is the key into policy and oracle tables.
RoundState is the full mutable game state used by BlackjackEnv.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from blackjack_opt.config import Rules
from blackjack_opt.hand import Hand


@dataclass(frozen=True)
class DecisionCell:
    """
    Minimal information needed to make an optimal decision.
    Hashable — used as dict key in policy and oracle tables.
    """
    player_total: int          # 4–21
    dealer_upcard: int         # 2–11
    is_soft: bool
    is_pair: bool              # True if exactly 2 cards of equal value
    pair_rank: int | None      # e.g. 8 for 8-8, 11 for A-A; None if not a pair
    can_double: bool           # precomputed from rules
    can_split: bool            # precomputed from rules
    from_split: bool           # affects BJ payout eligibility
    split_depth: int           # number of splits that created this hand (0 = original)

    def __str__(self) -> str:
        if self.is_pair:
            hand_str = f"Pair-{self.pair_rank}"
        elif self.is_soft:
            hand_str = f"Soft-{self.player_total}"
        else:
            hand_str = f"Hard-{self.player_total}"
        flags = []
        if self.can_double:   flags.append("D")
        if self.can_split:    flags.append("P")
        if self.from_split:   flags.append("spl")
        flag_str = f"[{','.join(flags)}]" if flags else ""
        return f"{hand_str} vs {self.dealer_upcard}{flag_str}"


def make_cell(
    hand: Hand,
    dealer_upcard: int,
    rules: Rules,
    n_hands_active: int,
    split_depth: int,
) -> DecisionCell:
    """Construct a DecisionCell from live game state."""
    from blackjack_opt.rules import can_double as _can_double, can_split as _can_split
    return DecisionCell(
        player_total  = hand.total,
        dealer_upcard = dealer_upcard,
        is_soft       = hand.is_soft,
        is_pair       = hand.pair_rank is not None,
        pair_rank     = hand.pair_rank,
        can_double    = _can_double(hand, rules),
        can_split     = _can_split(hand, rules, n_hands_active),
        from_split    = hand.from_split,
        split_depth   = split_depth,
    )


@dataclass
class RoundState:
    """Full mutable state for one round of blackjack."""
    hands: list[Hand]
    current_hand_idx: int
    dealer_upcard: int
    dealer_hole: int
    bet: float
    hand_bets: list[float]         # per-hand bet (may be doubled)
    hand_split_depths: list[int]   # split depth per hand
    rules: Rules
    rng: np.random.Generator
    done: bool = False
    reward: float = 0.0
    n_splits: int = 0              # total splits performed this round

    @property
    def current_hand(self) -> Hand:
        return self.hands[self.current_hand_idx]

    @property
    def n_hands(self) -> int:
        return len(self.hands)
