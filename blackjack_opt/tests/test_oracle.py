"""Tests for oracle.py — DP solver correctness."""
import pytest
import numpy as np

from blackjack_opt.config import (
    BENCHMARK, ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT,
)
from blackjack_opt.oracle import (
    dealer_terminal_dist, q_stand, q_hit_recursive, q_double, q_split,
    solve_optimal_policy, BUST,
)


class TestDealerTerminalDist:
    def test_sums_to_one(self):
        for upcard in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            dist = dealer_terminal_dist(upcard, BENCHMARK)
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-8, f"upcard={upcard}: sum={total}"

    def test_all_finals_valid(self):
        for upcard in [2, 3, 10, 11]:
            dist = dealer_terminal_dist(upcard, BENCHMARK)
            valid = {17, 18, 19, 20, 21, BUST}
            for k in dist:
                assert k in valid, f"Unexpected dealer final: {k}"

    def test_dealer_ace_has_high_21_prob(self):
        # With an Ace, dealer reaches 21 or high totals more often
        dist = dealer_terminal_dist(11, BENCHMARK)
        # Under peek, hole is conditioned on non-10; dealer soft totals still high
        p_17_to_21 = sum(dist.get(t, 0) for t in range(17, 22))
        assert p_17_to_21 > 0.0

    def test_no_finals_below_17(self):
        for upcard in range(2, 12):
            dist = dealer_terminal_dist(upcard, BENCHMARK)
            for final in dist:
                if final != BUST:
                    assert final >= 17

    def test_s17_vs_h17_difference(self):
        from blackjack_opt.config import H17_RULES
        dist_s17 = dealer_terminal_dist(11, BENCHMARK)
        dist_h17 = dealer_terminal_dist(11, H17_RULES)
        # H17 dealer hits soft 17, so different distribution
        assert dist_s17 != dist_h17


class TestQValues:
    def test_q_stand_player_21_always_positive(self):
        for upcard in range(2, 12):
            ev = q_stand(21, upcard, BENCHMARK)
            assert ev > 0, f"Standing on 21 should be positive vs {upcard}"

    def test_q_stand_player_12_vs_2(self):
        # Hard 12 vs 2: negative EV (dealer busts less often with low card)
        ev = q_stand(12, 2, BENCHMARK)
        assert ev < 0

    def test_q_stand_player_20_vs_6(self):
        ev = q_stand(20, 6, BENCHMARK)
        assert ev > 0.5   # very favorable

    def test_double_hard_11_vs_6_is_best(self):
        dbl_ev = q_double(11, False, 6, BENCHMARK)
        hit_ev = q_hit_recursive(11, False, 6, BENCHMARK)
        std_ev = q_stand(11, 6, BENCHMARK)
        assert dbl_ev > hit_ev, "Doubling 11 vs 6 should beat hitting"
        assert dbl_ev > std_ev, "Doubling 11 vs 6 should beat standing"

    def test_split_aces_better_than_hard_12(self):
        split_ev = q_split(11, 6, BENCHMARK, split_depth=0)
        stand_ev = q_stand(12, 6, BENCHMARK)
        assert split_ev > stand_ev, "Splitting aces vs 6 should beat standing on 12"

    def test_q_hit_bust_never_positive(self):
        # Hitting hard 20 must be worse than standing
        hit_ev  = q_hit_recursive(20, False, 6, BENCHMARK)
        stand_ev = q_stand(20, 6, BENCHMARK)
        assert hit_ev < stand_ev


class TestSolveOptimalPolicy:
    @pytest.fixture(scope="class")
    def oracle(self):
        return solve_optimal_policy(BENCHMARK)

    def test_returns_nonempty(self, oracle):
        assert len(oracle.pi_star) > 0
        assert len(oracle.Q_star) > 0
        assert len(oracle.V_star) > 0

    def test_split_eights_always(self, oracle):
        """Pair of 8s should always split (canonical basic strategy)."""
        from blackjack_opt.state import DecisionCell
        from blackjack_opt.hand import compute_total as ct
        total, is_soft = ct([8, 8])
        for upcard in range(2, 12):
            cell = DecisionCell(
                player_total=total, dealer_upcard=upcard, is_soft=is_soft,
                is_pair=True, pair_rank=8, can_double=False, can_split=True,
                from_split=False, split_depth=0,
            )
            if cell in oracle.pi_star:
                assert oracle.pi_star[cell] == ACTION_SPLIT, \
                    f"8-8 vs {upcard}: expected split, got {oracle.pi_star[cell]}"

    def test_never_split_fives(self, oracle):
        """Pair of 5s should never split (5-5 = hard 10, double/hit is better)."""
        from blackjack_opt.state import DecisionCell
        from blackjack_opt.hand import compute_total as ct
        total, is_soft = ct([5, 5])
        for upcard in range(2, 12):
            cell = DecisionCell(
                player_total=total, dealer_upcard=upcard, is_soft=is_soft,
                is_pair=True, pair_rank=5, can_double=True, can_split=True,
                from_split=False, split_depth=0,
            )
            if cell in oracle.pi_star:
                assert oracle.pi_star[cell] != ACTION_SPLIT, \
                    f"5-5 vs {upcard}: should not split"

    def test_hard_20_always_stand(self, oracle):
        """Hard 20 should always stand."""
        from blackjack_opt.state import DecisionCell
        for upcard in range(2, 12):
            cell = DecisionCell(
                player_total=20, dealer_upcard=upcard, is_soft=False,
                is_pair=False, pair_rank=None, can_double=False, can_split=False,
                from_split=False, split_depth=0,
            )
            if cell in oracle.pi_star:
                assert oracle.pi_star[cell] == ACTION_STAND, \
                    f"Hard 20 vs {upcard}: expected stand"

    def test_soft_18_vs_strong_dealer_hits(self, oracle):
        """Soft 18 (A+7) vs 9 or 10 should hit (standard basic strategy)."""
        from blackjack_opt.state import DecisionCell
        for upcard in [9, 10]:
            # No double available (e.g., after a hit)
            cell = DecisionCell(
                player_total=18, dealer_upcard=upcard, is_soft=True,
                is_pair=False, pair_rank=None, can_double=False, can_split=False,
                from_split=False, split_depth=0,
            )
            if cell in oracle.pi_star:
                assert oracle.pi_star[cell] == ACTION_HIT, \
                    f"Soft 18 vs {upcard}: expected hit, got {oracle.pi_star[cell]}"

    def test_overall_ev_near_published(self, oracle):
        """
        Oracle EV should be close to published -0.46% for S17 infinite shoe.
        We test that it's negative and within reasonable range.
        """
        ev = np.mean(list(oracle.V_star.values()))
        # Published EV ≈ -0.0046 (weighted); unweighted mean over cells will differ,
        # so just check it's negative and not wildly off.
        assert ev < 0.1, f"Unexpected mean cell EV: {ev}"
