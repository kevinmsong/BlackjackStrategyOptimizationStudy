"""Tests for hand.py — total computation, soft/hard, blackjack detection."""
import pytest
from blackjack_opt.hand import Hand, compute_total


class TestComputeTotal:
    def test_hard_total_basic(self):
        total, soft = compute_total([10, 6])
        assert total == 16
        assert soft is False

    def test_soft_total_basic(self):
        total, soft = compute_total([11, 6])   # A + 6
        assert total == 17
        assert soft is True

    def test_soft_total_reduction_one_ace(self):
        total, soft = compute_total([11, 6, 8])   # A + 6 + 8 = 25 → 15
        assert total == 15
        assert soft is False

    def test_double_ace(self):
        total, soft = compute_total([11, 11])   # A + A = 22 → 12
        assert total == 12
        assert soft is True

    def test_triple_ace(self):
        total, soft = compute_total([11, 11, 11])  # 33 → 23 → 13
        assert total == 13
        assert soft is True

    def test_bust(self):
        total, soft = compute_total([10, 10, 5])
        assert total == 25
        assert soft is False

    def test_soft_21(self):
        total, soft = compute_total([11, 10])   # A + 10 = 21
        assert total == 21
        assert soft is True

    def test_ace_reduces_to_hard(self):
        total, soft = compute_total([11, 7, 7])   # A + 7 + 7 = 25 → 15
        assert total == 15
        assert soft is False

    def test_hard_17(self):
        total, soft = compute_total([10, 7])
        assert total == 17
        assert soft is False


class TestHandFromCards:
    def test_blackjack_detection(self):
        h = Hand.from_cards([11, 10])
        assert h.is_blackjack is True
        assert h.total == 21

    def test_blackjack_from_split_not_blackjack(self):
        h = Hand.from_cards([11, 10], from_split=True)
        assert h.is_blackjack is False
        assert h.total == 21

    def test_pair_detection(self):
        h = Hand.from_cards([8, 8])
        assert h.pair_rank == 8   # pair_rank set when 2 equal-value cards

    def test_pair_of_aces(self):
        h = Hand.from_cards([11, 11])
        assert h.pair_rank == 11
        assert h.total == 12

    def test_pair_of_tens(self):
        # T+J both value 10 → pair
        h = Hand.from_cards([10, 10])
        assert h.pair_rank == 10

    def test_no_pair_different_values(self):
        h = Hand.from_cards([7, 9])
        assert h.pair_rank is None

    def test_bust_flag(self):
        h = Hand.from_cards([10, 10, 5])
        assert h.is_busted is True
        assert h.resolved is True

    def test_not_busted(self):
        h = Hand.from_cards([10, 9])
        assert h.is_busted is False

    def test_soft_total_on_hand(self):
        h = Hand.from_cards([11, 6])
        assert h.is_soft is True
        assert h.total == 17

    def test_split_aces_resolved(self):
        h = Hand.from_cards([11, 7], from_split=True, split_aces=True)
        assert h.resolved is True
        assert h.split_aces is True


class TestHandAddCard:
    def test_add_card_basic(self):
        h = Hand.from_cards([10, 6])
        h.add_card(5)
        assert h.total == 21
        assert h.n_cards == 3

    def test_add_card_busts(self):
        h = Hand.from_cards([10, 9])
        h.add_card(5)
        assert h.is_busted is True
        assert h.resolved is True

    def test_add_card_soft_becomes_hard(self):
        h = Hand.from_cards([11, 7])   # soft 18
        h.add_card(7)                  # soft 25 → hard 15
        assert h.total == 15
        assert h.is_soft is False
        assert h.is_busted is False

    def test_add_card_removes_pair(self):
        h = Hand.from_cards([8, 8])
        assert h.pair_rank == 8
        h.add_card(5)
        assert h.pair_rank is None

    def test_add_card_double_ace(self):
        h = Hand.from_cards([11, 11])  # total 12, soft
        h.add_card(9)                  # 12 + 9 = 21
        assert h.total == 21
        assert not h.is_busted
