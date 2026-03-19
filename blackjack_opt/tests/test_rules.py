"""Tests for rules.py — action legality predicates."""
import numpy as np
import pytest

from blackjack_opt.config import BENCHMARK, S17_NO_DAS, S17_SURRENDER, NUM_ACTIONS
from blackjack_opt.hand import Hand
from blackjack_opt.rules import (
    can_hit, can_stand, can_double, can_split, can_surrender,
    legal_actions, double_allowed_for_total,
)


def make_hand(cards, from_split=False, split_aces=False, doubled=False, resolved=False):
    h = Hand.from_cards(cards, from_split=from_split, split_aces=split_aces)
    h.doubled = doubled
    h.resolved = resolved
    return h


class TestDoubleAllowed:
    def test_any2(self):
        assert double_allowed_for_total(13, True, "any2") is True
        assert double_allowed_for_total(7, False, "any2") is True

    def test_9_10_11_hard_only(self):
        assert double_allowed_for_total(9, False, "9_10_11") is True
        assert double_allowed_for_total(10, False, "9_10_11") is True
        assert double_allowed_for_total(11, False, "9_10_11") is True
        assert double_allowed_for_total(8, False, "9_10_11") is False
        assert double_allowed_for_total(9, True, "9_10_11") is False   # soft 9 = not valid

    def test_10_11_only(self):
        assert double_allowed_for_total(10, False, "10_11") is True
        assert double_allowed_for_total(11, False, "10_11") is True
        assert double_allowed_for_total(9, False, "10_11") is False


class TestCanHit:
    def test_can_hit_normal(self):
        h = make_hand([10, 6])
        assert can_hit(h, BENCHMARK) is True

    def test_cannot_hit_busted(self):
        h = make_hand([10, 10, 5])
        assert can_hit(h, BENCHMARK) is False

    def test_cannot_hit_doubled(self):
        h = make_hand([10, 6])
        h.doubled = True
        assert can_hit(h, BENCHMARK) is False

    def test_cannot_hit_split_aces(self):
        h = make_hand([11, 7], from_split=True, split_aces=True)
        assert can_hit(h, BENCHMARK) is False

    def test_cannot_hit_resolved(self):
        h = make_hand([10, 8])
        h.resolved = True
        assert can_hit(h, BENCHMARK) is False


class TestCanDouble:
    def test_can_double_two_cards(self):
        h = make_hand([10, 6])
        assert can_double(h, BENCHMARK) is True

    def test_cannot_double_three_cards(self):
        h = make_hand([5, 6, 5])
        assert can_double(h, BENCHMARK) is False

    def test_das_allowed(self):
        h = make_hand([6, 5], from_split=True)
        assert can_double(h, BENCHMARK) is True   # BENCHMARK has DAS

    def test_das_not_allowed(self):
        h = make_hand([6, 5], from_split=True)
        assert can_double(h, S17_NO_DAS) is False

    def test_split_aces_no_double(self):
        h = make_hand([11, 10], from_split=True, split_aces=True)
        assert can_double(h, BENCHMARK) is False

    def test_double_rule_9_10_11(self):
        from blackjack_opt.config import Rules, RuleVariant
        rules = Rules(
            variant=RuleVariant.VEGAS_S17,
            blackjack_payout=1.5, dealer_stands_soft17=True, dealer_peek=True,
            double_allowed="9_10_11", double_after_split=True,
            resplit_max_hands=4, split_aces_one_card=True,
            surrender_allowed=False, insurance_allowed=False, infinite_shoe=True,
        )
        h8 = make_hand([4, 4])   # hard 8
        h9 = make_hand([4, 5])   # hard 9
        assert can_double(h8, rules) is False
        assert can_double(h9, rules) is True


class TestCanSplit:
    def test_can_split_pair(self):
        h = make_hand([8, 8])
        assert can_split(h, BENCHMARK, n_hands_active=1) is True

    def test_cannot_split_not_pair(self):
        h = make_hand([8, 7])
        assert can_split(h, BENCHMARK, n_hands_active=1) is False

    def test_cannot_split_at_max_hands(self):
        h = make_hand([8, 8])
        assert can_split(h, BENCHMARK, n_hands_active=4) is False

    def test_cannot_split_three_cards(self):
        h = make_hand([4, 4, 4])
        assert can_split(h, BENCHMARK, n_hands_active=1) is False

    def test_cannot_split_aces_hand(self):
        h = make_hand([11, 7], from_split=True, split_aces=True)
        assert can_split(h, BENCHMARK, n_hands_active=2) is False


class TestCanSurrender:
    def test_can_surrender(self):
        h = make_hand([10, 6])
        assert can_surrender(h, S17_SURRENDER) is True

    def test_cannot_surrender_disabled(self):
        h = make_hand([10, 6])
        assert can_surrender(h, BENCHMARK) is False   # BENCHMARK no surrender

    def test_cannot_surrender_three_cards(self):
        h = make_hand([5, 6, 5])
        assert can_surrender(h, S17_SURRENDER) is False

    def test_cannot_surrender_from_split(self):
        h = make_hand([10, 6], from_split=True)
        assert can_surrender(h, S17_SURRENDER) is False


class TestLegalActions:
    def test_shape(self):
        h = make_hand([10, 6])
        mask = legal_actions(h, BENCHMARK, n_hands_active=1)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_standard_two_card_hand(self):
        h = make_hand([10, 6])
        mask = legal_actions(h, BENCHMARK, n_hands_active=1)
        assert bool(mask[0])    # stand
        assert bool(mask[1])    # hit
        assert bool(mask[2])    # double
        assert not mask[3]      # no split (not a pair)
        assert not mask[4]      # no surrender (disabled)

    def test_pair_can_split(self):
        h = make_hand([8, 8])
        mask = legal_actions(h, BENCHMARK, n_hands_active=1)
        assert bool(mask[3])    # split
