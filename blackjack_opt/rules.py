"""
rules.py — Pure predicate functions for action legality.

Called by BlackjackEnv.legal_actions() and by the oracle when
enumerating reachable states.  No mutable state.
"""
from __future__ import annotations
import numpy as np

from blackjack_opt.config import (
    Rules,
    ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT, ACTION_SURRENDER,
    NUM_ACTIONS,
)
from blackjack_opt.hand import Hand


def double_allowed_for_total(total: int, is_soft: bool, double_rule: str) -> bool:
    """Does this total qualify under the given double rule?"""
    if double_rule == "any2":
        return True
    # Hard totals only for 9_10_11 and 10_11
    if is_soft:
        return False
    if double_rule == "9_10_11":
        return total in (9, 10, 11)
    if double_rule == "10_11":
        return total in (10, 11)
    raise ValueError(f"Unknown double rule: {double_rule!r}")


def can_hit(hand: Hand, rules: Rules) -> bool:
    if hand.is_busted or hand.doubled or hand.split_aces or hand.resolved:
        return False
    return True


def can_stand(hand: Hand, rules: Rules) -> bool:
    if hand.is_busted:
        return False
    return True


def can_double(hand: Hand, rules: Rules) -> bool:
    if hand.n_cards != 2:
        return False
    if hand.is_busted:
        return False
    if hand.split_aces:
        return False
    if hand.from_split and not rules.double_after_split:
        return False
    return double_allowed_for_total(hand.total, hand.is_soft, rules.double_allowed)


def can_split(hand: Hand, rules: Rules, n_hands_active: int) -> bool:
    if hand.n_cards != 2:
        return False
    if hand.pair_rank is None:
        return False
    if hand.split_aces:
        return False
    if n_hands_active >= rules.resplit_max_hands:
        return False
    return True


def can_surrender(hand: Hand, rules: Rules) -> bool:
    if not rules.surrender_allowed:
        return False
    if hand.n_cards != 2:
        return False
    if hand.from_split:
        return False
    return True


def legal_actions(hand: Hand, rules: Rules, n_hands_active: int) -> np.ndarray:
    """
    Returns bool array of shape (NUM_ACTIONS,):
    [stand, hit, double, split, surrender]
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[ACTION_STAND]     = can_stand(hand, rules)
    mask[ACTION_HIT]       = can_hit(hand, rules)
    mask[ACTION_DOUBLE]    = can_double(hand, rules)
    mask[ACTION_SPLIT]     = can_split(hand, rules, n_hands_active)
    mask[ACTION_SURRENDER] = can_surrender(hand, rules)
    return mask
