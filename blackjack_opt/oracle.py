"""
oracle.py — Exact DP oracle solver for infinite-shoe blackjack.

Computes ground-truth Q-values and optimal policy via dynamic programming.
Handles dealer peek conditional distribution, split recursion with depth
limits, and all rule variants.
"""
from __future__ import annotations
import functools
from typing import NamedTuple

import numpy as np

from blackjack_opt.config import (
    Rules, BENCHMARK,
    ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT, ACTION_SURRENDER,
    NUM_ACTIONS,
)
from blackjack_opt.cards import CARD_VALUES, DRAW_PROBS
from blackjack_opt.hand import compute_total
from blackjack_opt.state import DecisionCell

# Sentinel for "dealer busted" in distributions
BUST = 99


# ═══════════════════════════════════════════════════════════════════════════
# Dealer terminal distribution
# ═══════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=None)
def _dealer_draw(total: int, is_soft: bool, s17: bool) -> dict[int, float]:
    """
    Recursively compute the probability distribution over dealer final totals
    starting from a given (total, is_soft) state, drawing from infinite shoe.

    Returns dict mapping final_total -> probability.
    BUST (99) is used as the bust key.
    """
    # Stand condition
    if total > 21:
        return {BUST: 1.0}
    if total >= 18:
        return {total: 1.0}
    if total == 17:
        if is_soft and not s17:
            pass   # H17: hit soft 17
        else:
            return {total: 1.0}

    result: dict[int, float] = {}
    for v, p in DRAW_PROBS.items():
        new_total, new_soft = compute_total_fast(total, is_soft, v)
        sub = _dealer_draw(new_total, new_soft, s17)
        for final, prob in sub.items():
            result[final] = result.get(final, 0.0) + p * prob
    return result


def compute_total_fast(old_total: int, old_soft: bool, new_card: int) -> tuple[int, bool]:
    """
    Fast incremental total update: add new_card to existing (old_total, old_soft).
    """
    total = old_total + new_card
    is_soft = old_soft or (new_card == 11)
    # Count how many aces are at 11
    n_aces_as_11 = (1 if old_soft else 0) + (1 if new_card == 11 else 0)
    while total > 21 and n_aces_as_11 > 0:
        total -= 10
        n_aces_as_11 -= 1
    is_soft = n_aces_as_11 > 0
    return total, is_soft


@functools.lru_cache(maxsize=None)
def dealer_terminal_dist(upcard: int, rules: Rules) -> dict[int, float]:
    """
    Full dealer terminal distribution conditioned on:
    - No dealer blackjack (under peek rules)
    - Starting from upcard visible card

    Returns dict: final_total -> probability, with BUST (99) for busted hands.
    """
    s17 = rules.dealer_stands_soft17

    # Enumerate hole cards, conditioned on no-BJ if peek
    hole_dist: dict[int, float] = {}
    if rules.dealer_peek:
        if upcard == 11:      # Ace up: hole cannot be 10 (would be BJ)
            for v, p in DRAW_PROBS.items():
                if v != 10:
                    hole_dist[v] = p
            total_p = sum(hole_dist.values())
            hole_dist = {v: p / total_p for v, p in hole_dist.items()}
        elif upcard == 10:    # 10-value up: hole cannot be Ace
            for v, p in DRAW_PROBS.items():
                if v != 11:
                    hole_dist[v] = p
            total_p = sum(hole_dist.values())
            hole_dist = {v: p / total_p for v, p in hole_dist.items()}
        else:
            hole_dist = dict(DRAW_PROBS)
    else:
        hole_dist = dict(DRAW_PROBS)

    # For each possible hole card, compute dealer's starting state
    result: dict[int, float] = {}
    for hole, p_hole in hole_dist.items():
        dealer_start, dealer_soft = compute_total([upcard, hole])
        sub = _dealer_draw(dealer_start, dealer_soft, s17)
        for final, p_final in sub.items():
            result[final] = result.get(final, 0.0) + p_hole * p_final
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Player Q-value functions
# ═══════════════════════════════════════════════════════════════════════════

def q_stand(player_total: int, dealer_upcard: int, rules: Rules) -> float:
    """EV of standing with player_total against dealer_upcard."""
    dist = dealer_terminal_dist(dealer_upcard, rules)
    ev = 0.0
    for final, prob in dist.items():
        if final == BUST:
            ev += prob * 1.0   # dealer busts → player wins
        elif player_total > final:
            ev += prob * 1.0
        elif player_total < final:
            ev += prob * -1.0
        # else push: 0
    return ev


def q_hit(
    player_total: int,
    is_soft: bool,
    dealer_upcard: int,
    rules: Rules,
    can_double_after: bool = False,
) -> float:
    """
    EV of hitting from (player_total, is_soft).
    After a hit, player can only hit or stand (not double/split/surrender).
    """
    ev = 0.0
    for v, p in DRAW_PROBS.items():
        new_total, new_soft = compute_total_fast(player_total, is_soft, v)
        if new_total > 21:
            ev += p * -1.0
        else:
            # After first hit: can only hit or stand
            best = max(
                q_stand(new_total, dealer_upcard, rules),
                q_hit_recursive(new_total, new_soft, dealer_upcard, rules),
            )
            ev += p * best
    return ev


@functools.lru_cache(maxsize=None)
def q_hit_recursive(
    player_total: int,
    is_soft: bool,
    dealer_upcard: int,
    rules: Rules,
) -> float:
    """
    EV of hitting from a mid-hand state (can only hit or stand next).
    Memoized because this is called repeatedly.
    """
    ev = 0.0
    for v, p in DRAW_PROBS.items():
        new_total, new_soft = compute_total_fast(player_total, is_soft, v)
        if new_total > 21:
            ev += p * -1.0
        else:
            best = max(
                q_stand(new_total, dealer_upcard, rules),
                q_hit_recursive(new_total, new_soft, dealer_upcard, rules),
            )
            ev += p * best
    return ev


def q_double(
    player_total: int,
    is_soft: bool,
    dealer_upcard: int,
    rules: Rules,
) -> float:
    """
    EV of doubling: take exactly one card then stand.
    Returns EV per original bet (net is 2× this because bet doubles).
    The caller should use 2*q_double as the actual dollar EV.
    We store it as 2× directly for comparison with other actions on unit-bet scale.
    """
    ev = 0.0
    for v, p in DRAW_PROBS.items():
        new_total, _ = compute_total_fast(player_total, is_soft, v)
        if new_total > 21:
            ev += p * -1.0   # bust after double
        else:
            ev += p * q_stand(new_total, dealer_upcard, rules)
    return 2.0 * ev   # bet is doubled


def q_surrender_val() -> float:
    """EV of surrendering: always -0.5."""
    return -0.5


# ─── Split EV ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def q_split(
    pair_rank: int,
    dealer_upcard: int,
    rules: Rules,
    split_depth: int,
) -> float:
    """
    EV of splitting a pair.
    split_depth: depth of the hand being split (0 = original hand).
    New child hands will be at depth split_depth+1.
    """
    new_depth = split_depth + 1
    is_ace_split = (pair_rank == 11)

    if is_ace_split:
        # Each child hand: ace + one random card, then forced stand
        # A + 2 → 13, A + 3 → 14, ..., A + 10 → 11 (reduced), A + A → 12
        ev_per_hand = 0.0
        for v, p in DRAW_PROBS.items():
            hand_total, hand_soft = compute_total([11, v])
            ev_per_hand += p * q_stand(hand_total, dealer_upcard, rules)
        return 2.0 * ev_per_hand

    # Non-ace split: each child hand starts with one card (pair_rank)
    # and can be played normally (hit, stand, double if DAS, re-split if allowed)
    ev_per_hand = _optimal_ev_single_card(
        first_card    = pair_rank,
        dealer_upcard = dealer_upcard,
        rules         = rules,
        split_depth   = new_depth,
        from_split    = True,
    )
    return 2.0 * ev_per_hand


@functools.lru_cache(maxsize=None)
def _optimal_ev_single_card(
    first_card: int,
    dealer_upcard: int,
    rules: Rules,
    split_depth: int,
    from_split: bool,
) -> float:
    """
    EV of playing a hand that starts with a single card (first_card),
    then receives one more card dealt immediately, from a split context.
    """
    ev = 0.0
    for v, p in DRAW_PROBS.items():
        # Two-card hand after split: (first_card, v)
        total, is_soft = compute_total([first_card, v])
        pair_rank = first_card if first_card == v else None

        n_hands_for_can_split = split_depth + 1   # approximate
        can_dbl = _can_double_dp(total, is_soft, from_split, rules)
        can_spl = (
            pair_rank is not None
            and not (first_card == 11)   # already handled as ace split
            and n_hands_for_can_split < rules.resplit_max_hands
        )

        best = q_stand(total, dealer_upcard, rules)

        # Hit
        hit_ev = q_hit_recursive(total, is_soft, dealer_upcard, rules)
        best = max(best, hit_ev)

        # Double (if allowed)
        if can_dbl:
            dbl_ev = q_double(total, is_soft, dealer_upcard, rules)
            best = max(best, dbl_ev)

        # Re-split (if allowed)
        if can_spl:
            resplit_ev = q_split(pair_rank, dealer_upcard, rules, split_depth)
            best = max(best, resplit_ev)

        ev += p * best
    return ev


def _can_double_dp(total: int, is_soft: bool, from_split: bool, rules: Rules) -> bool:
    if from_split and not rules.double_after_split:
        return False
    from blackjack_opt.rules import double_allowed_for_total
    return double_allowed_for_total(total, is_soft, rules.double_allowed)


# ═══════════════════════════════════════════════════════════════════════════
# Policy solver
# ═══════════════════════════════════════════════════════════════════════════

class OracleResult(NamedTuple):
    Q_star: dict[DecisionCell, np.ndarray]   # shape (NUM_ACTIONS,) per cell
    pi_star: dict[DecisionCell, int]          # optimal action per cell
    V_star: dict[DecisionCell, float]         # optimal value per cell


def solve_optimal_policy(rules: Rules = BENCHMARK) -> OracleResult:
    """
    Enumerate all canonical decision cells and compute exact Q-values.
    Returns OracleResult with Q_star, pi_star, V_star.
    """
    Q_star:  dict[DecisionCell, np.ndarray] = {}
    pi_star: dict[DecisionCell, int]        = {}
    V_star:  dict[DecisionCell, float]      = {}

    dealer_upcards = list(range(2, 12))   # 2..10 and Ace=11

    # ── Hard totals (non-pair): player total 4–21 ───────────────────────────
    # Enumerate both can_double=True and can_double=False for every combination.
    # can_double=False cells are reached after any hit (3+ card hands).
    for total in range(4, 22):
        for upcard in dealer_upcards:
            for split_depth in range(rules.resplit_max_hands):
                for from_split in [False, True]:
                    if split_depth > 0 and not from_split:
                        continue
                    for can_dbl in [True, False]:
                        # Respect rule: DAS off → from_split hands can't double
                        if can_dbl and from_split and not rules.double_after_split:
                            continue
                        cell = DecisionCell(
                            player_total  = total,
                            dealer_upcard = upcard,
                            is_soft       = False,
                            is_pair       = False,
                            pair_rank     = None,
                            can_double    = can_dbl,
                            can_split     = False,
                            from_split    = from_split,
                            split_depth   = split_depth,
                        )
                        q = _compute_q(cell, rules)
                        Q_star[cell]  = q
                        best_action   = int(np.argmax(q))
                        pi_star[cell] = best_action
                        V_star[cell]  = float(q[best_action])

    # ── Soft totals (non-pair): 12–21 ───────────────────────────────────────
    for total in range(12, 22):
        for upcard in dealer_upcards:
            for split_depth in range(rules.resplit_max_hands):
                for from_split in [False, True]:
                    if split_depth > 0 and not from_split:
                        continue
                    for can_dbl in [True, False]:
                        if can_dbl and from_split and not rules.double_after_split:
                            continue
                        cell = DecisionCell(
                            player_total  = total,
                            dealer_upcard = upcard,
                            is_soft       = True,
                            is_pair       = False,
                            pair_rank     = None,
                            can_double    = can_dbl,
                            can_split     = False,
                            from_split    = from_split,
                            split_depth   = split_depth,
                        )
                        q = _compute_q(cell, rules)
                        Q_star[cell]  = q
                        best_action   = int(np.argmax(q))
                        pi_star[cell] = best_action
                        V_star[cell]  = float(q[best_action])

    # ── Pairs ────────────────────────────────────────────────────────────────
    for pair_rank in range(2, 12):   # 2..10, Ace=11
        total, is_soft = compute_total([pair_rank, pair_rank])
        for upcard in dealer_upcards:
            for split_depth in range(rules.resplit_max_hands):
                for from_split in [False, True]:
                    if split_depth > 0 and not from_split:
                        continue
                    for can_dbl in [True, False]:
                        if can_dbl and from_split and not rules.double_after_split:
                            continue
                        for can_spl in [True, False]:
                            # can_spl=True only makes sense when depth allows re-split
                            if can_spl and (split_depth + 1) >= rules.resplit_max_hands:
                                continue
                            cell = DecisionCell(
                                player_total  = total,
                                dealer_upcard = upcard,
                                is_soft       = is_soft,
                                is_pair       = True,
                                pair_rank     = pair_rank,
                                can_double    = can_dbl,
                                can_split     = can_spl,
                                from_split    = from_split,
                                split_depth   = split_depth,
                            )
                            q = _compute_q(cell, rules)
                            Q_star[cell]  = q
                            best_action   = int(np.argmax(q))
                            pi_star[cell] = best_action
                            V_star[cell]  = float(q[best_action])

    return OracleResult(Q_star=Q_star, pi_star=pi_star, V_star=V_star)


def _compute_q(cell: DecisionCell, rules: Rules) -> np.ndarray:
    """Compute Q-vector for a cell; illegal actions get -inf."""
    q = np.full(NUM_ACTIONS, -np.inf)
    total   = cell.player_total
    is_soft = cell.is_soft
    upcard  = cell.dealer_upcard
    depth   = cell.split_depth

    # Stand is always legal unless busted (busted cells not in our enumeration)
    q[ACTION_STAND] = q_stand(total, upcard, rules)

    # Hit
    q[ACTION_HIT] = q_hit_recursive(total, is_soft, upcard, rules)

    # Double
    if cell.can_double:
        q[ACTION_DOUBLE] = q_double(total, is_soft, upcard, rules)

    # Split
    if cell.can_split and cell.pair_rank is not None:
        q[ACTION_SPLIT] = q_split(cell.pair_rank, upcard, rules, depth)

    # Surrender
    if rules.surrender_allowed and not cell.from_split and depth == 0:
        q[ACTION_SURRENDER] = q_surrender_val()

    return q
