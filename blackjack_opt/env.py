"""
env.py — BlackjackEnv: the simulation engine.

Exposes reset / legal_actions / step / resolve_dealer interface.
All three optimizers interact with the environment identically.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from blackjack_opt.config import (
    Rules, BENCHMARK,
    ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT, ACTION_SURRENDER,
)
from blackjack_opt.cards import draw_card_value
from blackjack_opt.hand import Hand, compute_total
from blackjack_opt.rules import legal_actions as _legal_actions
from blackjack_opt.state import DecisionCell, RoundState, make_cell


class BlackjackEnv:
    def __init__(
        self,
        rules: Rules = BENCHMARK,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rules = rules
        self.rng   = rng or np.random.default_rng()

    # ── Public interface ─────────────────────────────────────────────────────

    def reset(self, bet: float = 1.0) -> tuple[RoundState, Optional[DecisionCell]]:
        """
        Deal a fresh hand.  Returns (state, first_cell).
        first_cell is None if the round is over immediately
        (e.g. dealer blackjack under peek rules).
        """
        state = self._deal_initial(bet)
        if state.done:
            return state, None
        cell = self._current_cell(state)
        return state, cell

    def legal_actions(self, state: RoundState) -> np.ndarray:
        hand = state.current_hand
        return _legal_actions(hand, self.rules, state.n_hands)

    def step(
        self,
        state: RoundState,
        action: int,
    ) -> tuple[RoundState, Optional[DecisionCell], float, bool]:
        """
        Apply action to current hand.  Returns (state, next_cell, reward, done).
        reward is 0.0 until done=True, then total net chips won/lost.
        """
        assert not state.done, "Cannot step a finished round"

        if action == ACTION_STAND:
            state.current_hand.resolved = True
            return self._advance(state)

        elif action == ACTION_HIT:
            v = draw_card_value(state.rng)
            state.current_hand.add_card(v)
            if state.current_hand.is_busted:
                state.current_hand.resolved = True
                return self._advance(state)
            return state, self._current_cell(state), 0.0, False

        elif action == ACTION_DOUBLE:
            v = draw_card_value(state.rng)
            state.current_hand.add_card(v)
            state.current_hand.doubled = True
            state.current_hand.resolved = True
            state.hand_bets[state.current_hand_idx] *= 2.0
            return self._advance(state)

        elif action == ACTION_SPLIT:
            return self._apply_split(state)

        elif action == ACTION_SURRENDER:
            # Lose half the bet; round ends for this hand
            state.done = True
            state.reward = -0.5 * state.bet
            return state, None, state.reward, True

        else:
            raise ValueError(f"Unknown action: {action}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _deal_initial(self, bet: float) -> RoundState:
        rng   = self.rng
        rules = self.rules

        # Draw player's two cards, then dealer upcard, then dealer hole
        p1 = draw_card_value(rng)
        p2 = draw_card_value(rng)
        dealer_up = draw_card_value(rng)

        # Hole card always drawn unconditionally.
        # Dealer BJ is resolved as a round-ending event below (peek check).
        # The oracle conditions on no-BJ only for DP value computation;
        # the simulation must see all rounds including dealer-BJ losses.
        dealer_hole = draw_card_value(rng)

        player_hand = Hand.from_cards([p1, p2])
        state = RoundState(
            hands              = [player_hand],
            current_hand_idx   = 0,
            dealer_upcard      = dealer_up,
            dealer_hole        = dealer_hole,
            bet                = bet,
            hand_bets          = [bet],
            hand_split_depths  = [0],
            rules              = rules,
            rng                = rng,
        )

        # Check dealer blackjack under peek
        dealer_total, _ = compute_total([dealer_up, dealer_hole])
        if rules.dealer_peek and dealer_total == 21:
            # Dealer has blackjack
            state.done = True
            if player_hand.is_blackjack:
                state.reward = 0.0   # push
            else:
                state.reward = -bet  # loss
            return state

        # Check player blackjack (only if dealer doesn't have one)
        if player_hand.is_blackjack:
            state.done = True
            state.reward = bet * rules.blackjack_payout
            return state

        return state

    def _apply_split(self, state: RoundState) -> tuple[RoundState, Optional[DecisionCell], float, bool]:
        idx  = state.current_hand_idx
        hand = state.hands[idx]
        rng  = state.rng

        assert hand.pair_rank is not None
        v1, v2 = hand.cards[0], hand.cards[1]
        is_ace_split = (v1 == 11)

        state.n_splits += 1
        new_depth = state.hand_split_depths[idx] + 1

        # Draw one card for each new hand immediately
        c1 = draw_card_value(rng)
        c2 = draw_card_value(rng)

        h1 = Hand.from_cards([v1, c1], from_split=True, split_aces=is_ace_split)
        h2 = Hand.from_cards([v2, c2], from_split=True, split_aces=is_ace_split)

        # Replace original hand with two new hands
        state.hands[idx:idx+1] = [h1, h2]
        state.hand_bets[idx:idx+1] = [state.bet, state.bet]
        state.hand_split_depths[idx:idx+1] = [new_depth, new_depth]

        # Split aces: both hands immediately resolved (one card only)
        if is_ace_split:
            h1.resolved = True
            h2.resolved = True
            state.current_hand_idx = idx
            return self._advance(state)

        # Otherwise play the first split hand
        state.current_hand_idx = idx
        return state, self._current_cell(state), 0.0, False

    def _advance(self, state: RoundState) -> tuple[RoundState, Optional[DecisionCell], float, bool]:
        """
        Find next hand needing a decision; if all resolved, run dealer and settle.
        """
        # Search for next unresolved hand
        for i in range(len(state.hands)):
            if not state.hands[i].resolved:
                state.current_hand_idx = i
                return state, self._current_cell(state), 0.0, False

        # All hands resolved — run dealer
        reward = self._resolve_dealer(state)
        state.done   = True
        state.reward = reward
        return state, None, reward, True

    def _resolve_dealer(self, state: RoundState) -> float:
        """
        Draw dealer cards to completion (S17 or H17) then settle all hands.
        Returns total net reward (sum over all hands).
        """
        rules = self.rules
        rng   = state.rng

        # Build dealer hand starting from upcard + hole
        dealer_cards = [state.dealer_upcard, state.dealer_hole]
        dtotal, dsoft = compute_total(dealer_cards)

        while True:
            # Dealer stands on hard 17+, or soft 17+ under S17
            if dtotal >= 18:
                break
            if dtotal == 17:
                if dsoft and not rules.dealer_stands_soft17:
                    pass   # H17: hit soft 17
                else:
                    break  # S17 or hard 17: stand
            v = draw_card_value(rng)
            dealer_cards.append(v)
            dtotal, dsoft = compute_total(dealer_cards)

        dealer_busted = dtotal > 21

        total_reward = 0.0
        for i, hand in enumerate(state.hands):
            bet = state.hand_bets[i]
            total_reward += self._settle_hand(hand, dtotal, dealer_busted, bet)
        return total_reward

    def _settle_hand(
        self,
        hand: Hand,
        dealer_total: int,
        dealer_busted: bool,
        bet: float,
    ) -> float:
        if hand.is_busted:
            return -bet
        if hand.is_blackjack:
            return bet * self.rules.blackjack_payout
        if dealer_busted:
            return bet
        if hand.total > dealer_total:
            return bet
        if hand.total < dealer_total:
            return -bet
        return 0.0   # push

    def _current_cell(self, state: RoundState) -> DecisionCell:
        return make_cell(
            hand          = state.current_hand,
            dealer_upcard = state.dealer_upcard,
            rules         = state.rules,
            n_hands_active= state.n_hands,
            split_depth   = state.hand_split_depths[state.current_hand_idx],
        )
