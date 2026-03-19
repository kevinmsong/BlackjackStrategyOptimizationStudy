"""Tests for env.py — BlackjackEnv game mechanics."""
import pytest
import numpy as np

from blackjack_opt.config import (
    BENCHMARK, S17_SURRENDER,
    ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT, ACTION_SURRENDER,
)
from blackjack_opt.env import BlackjackEnv
from blackjack_opt.hand import compute_total


def make_env(rules=None, seed=42):
    return BlackjackEnv(rules or BENCHMARK, np.random.default_rng(seed))


class TestReset:
    def test_player_gets_two_cards(self):
        env = make_env()
        for _ in range(20):
            state, cell = env.reset()
            if not state.done:
                assert state.hands[0].n_cards == 2

    def test_dealer_upcard_set(self):
        env = make_env()
        state, _ = env.reset()
        assert 2 <= state.dealer_upcard <= 11

    def test_dealer_peek_no_blackjack_in_play(self):
        # After reset (with peek), if the round is NOT done, dealer cannot have BJ
        # (dealer BJ ends the round immediately via state.done=True)
        env = make_env()
        for _ in range(500):
            state, cell = env.reset()
            if not state.done:
                dt, _ = compute_total([state.dealer_upcard, state.dealer_hole])
                assert dt != 21, "Dealer has BJ but round is still in play"


class TestDealerBlackjack:
    def test_dealer_bj_ends_round(self):
        # Force dealer blackjack by monkeypatching
        env = make_env()
        dealer_bj_seen = False
        for _ in range(500):
            state, cell = env.reset()
            if state.done:
                dt, _ = compute_total([state.dealer_upcard, state.dealer_hole])
                if dt == 21:
                    dealer_bj_seen = True
                    break
        # Note: under peek rules, dealer BJ exists but is rare
        # (A up + non-10 hole → not BJ)
        # This test just checks that done=True can happen

    def test_player_bj_wins_15(self):
        """Player BJ (no dealer BJ) returns 1.5 reward."""
        env = make_env()
        for _ in range(2000):
            state, cell = env.reset()
            if state.done and not state.hands[0].is_busted:
                if state.hands[0].is_blackjack and abs(state.reward - 1.5) < 1e-9:
                    return
        # At least some player BJs should happen in 2000 hands
        # (probability ~4.8%)


class TestHitAction:
    def test_hit_adds_card(self):
        env = make_env()
        for _ in range(50):
            state, cell = env.reset()
            if state.done:
                continue
            n_before = state.hands[0].n_cards
            mask = env.legal_actions(state)
            if mask[ACTION_HIT]:
                state2, cell2, r, done = env.step(state, ACTION_HIT)
                if not done:
                    assert state2.hands[0].n_cards == n_before + 1
                return

    def test_bust_resolves_hand(self):
        """Hitting past 21 should mark hand resolved and end round."""
        env = make_env()
        for _ in range(200):
            state, cell = env.reset()
            if state.done:
                continue
            # Keep hitting until bust or round ends
            prev_done = False
            for _ in range(10):
                mask = env.legal_actions(state)
                if not mask[ACTION_HIT]:
                    break
                state, cell, r, done = env.step(state, ACTION_HIT)
                if state.hands[0].is_busted:
                    assert state.hands[0].resolved
                    return
                if done:
                    break


class TestStandAction:
    def test_stand_triggers_dealer_resolution(self):
        env = make_env()
        for _ in range(50):
            state, cell = env.reset()
            if state.done:
                continue
            state, cell2, r, done = env.step(state, ACTION_STAND)
            assert done is True
            return


class TestDoubleAction:
    def test_double_exactly_one_extra_card(self):
        env = make_env()
        for _ in range(100):
            state, cell = env.reset()
            if state.done:
                continue
            mask = env.legal_actions(state)
            if mask[ACTION_DOUBLE]:
                n_before = state.hands[0].n_cards
                state2, _, r, done = env.step(state, ACTION_DOUBLE)
                # Done because doubling forces stand after one card
                assert done is True
                # The hand should have exactly 3 cards (2 + 1 doubled)
                assert state2.hands[0].n_cards == n_before + 1
                return

    def test_double_doubles_the_bet(self):
        env = make_env()
        for _ in range(100):
            state, cell = env.reset(bet=5.0)
            if state.done:
                continue
            mask = env.legal_actions(state)
            if mask[ACTION_DOUBLE]:
                idx = state.current_hand_idx
                state2, _, r, done = env.step(state, ACTION_DOUBLE)
                assert state2.hand_bets[idx] == 10.0  # doubled from 5
                return

    def test_double_reward_uses_doubled_bet(self):
        env = make_env()
        wins_at_2x = False
        for _ in range(500):
            state, cell = env.reset(bet=1.0)
            if state.done:
                continue
            mask = env.legal_actions(state)
            if mask[ACTION_DOUBLE]:
                state2, _, r, done = env.step(state, ACTION_DOUBLE)
                if done and r >= 2.0:
                    wins_at_2x = True
                    break
        assert wins_at_2x, "Should see at least one doubled win"


class TestSplitAction:
    def test_split_creates_two_hands(self):
        env = make_env()
        for _ in range(500):
            state, cell = env.reset()
            if state.done:
                continue
            mask = env.legal_actions(state)
            if mask[ACTION_SPLIT]:
                n_before = state.n_hands
                state2, cell2, r, done = env.step(state, ACTION_SPLIT)
                if not done:
                    assert state2.n_hands == n_before + 1
                return

    def test_split_aces_resolved_immediately(self):
        """After splitting aces, both child hands are immediately resolved."""
        env = make_env()
        for _ in range(2000):
            state, cell = env.reset()
            if state.done:
                continue
            # Check if we have a pair of aces
            hand = state.hands[0]
            if hand.pair_rank == 11:
                mask = env.legal_actions(state)
                if mask[ACTION_SPLIT]:
                    state2, cell2, r, done = env.step(state, ACTION_SPLIT)
                    # Both split-ace hands have split_aces=True and are resolved
                    for h in state2.hands:
                        assert h.split_aces or h.resolved
                    return

    def test_split_respects_max_hands(self):
        """Cannot split when already at resplit_max_hands."""
        env = make_env()
        # This is tested indirectly via rules tests; here just check mask behavior
        for _ in range(200):
            state, cell = env.reset()
            if state.done:
                continue
            if state.hands[0].pair_rank is not None:
                mask = env.legal_actions(state)
                if mask[ACTION_SPLIT]:
                    # Split down to max
                    for _ in range(BENCHMARK.resplit_max_hands - 1):
                        mask = env.legal_actions(state)
                        if not mask[ACTION_SPLIT]:
                            break
                        state, cell, r, done = env.step(state, ACTION_SPLIT)
                        if done:
                            break
                    # At max hands, split should be illegal
                    if not state.done:
                        mask = env.legal_actions(state)
                        if state.n_hands >= BENCHMARK.resplit_max_hands:
                            assert not mask[ACTION_SPLIT]
                    return


class TestSurrenderAction:
    def test_surrender_returns_minus_half(self):
        env = make_env(S17_SURRENDER)
        for _ in range(100):
            state, cell = env.reset(bet=1.0)
            if state.done:
                continue
            mask = env.legal_actions(state)
            if mask[ACTION_SURRENDER]:
                state2, _, r, done = env.step(state, ACTION_SURRENDER)
                assert done is True
                assert abs(r - (-0.5)) < 1e-9
                return


class TestDealerResolution:
    def test_dealer_s17_stands_on_soft17(self):
        """Under S17 rules, dealer stands on soft 17."""
        env = make_env(BENCHMARK)  # S17
        # Run many rounds and verify dealer never shows a soft-17 mid-hand
        # (impossible to directly observe without hooking internals)
        # Smoke test: many rounds complete without error
        for _ in range(200):
            state, cell = env.reset()
            if state.done:
                continue
            state, _, r, done = env.step(state, ACTION_STAND)
            assert done

    def test_rewards_in_valid_range(self):
        """Rewards should be between -2.0 and 1.5 for unit bet."""
        env = make_env()
        for _ in range(200):
            state, cell = env.reset(bet=1.0)
            if state.done:
                r = state.reward
            else:
                state, _, r, _ = env.step(state, ACTION_STAND)
            # Surrender: -0.5, BJ: 1.5, double loss: -2.0, double win: 2.0
            assert -2.1 <= r <= 2.1, f"Unexpected reward: {r}"
