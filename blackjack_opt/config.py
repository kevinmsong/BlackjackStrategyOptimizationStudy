"""
config.py — Frozen rule sets and game configuration.
Single source of truth imported by all downstream modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class RuleVariant(Enum):
    VEGAS_S17        = auto()   # benchmark frozen ruleset
    VEGAS_H17        = auto()
    S17_SURRENDER    = auto()
    S17_NO_DAS       = auto()
    H17_SURRENDER    = auto()


@dataclass(frozen=True)
class Rules:
    variant: RuleVariant
    blackjack_payout: float      # 1.5 for 3:2
    dealer_stands_soft17: bool   # True = S17, False = H17
    dealer_peek: bool
    double_allowed: str          # "any2" | "9_10_11" | "10_11"
    double_after_split: bool     # DAS
    resplit_max_hands: int       # max total hands (4 = up to 3 splits)
    split_aces_one_card: bool    # aces get one card only after split
    surrender_allowed: bool      # late surrender
    insurance_allowed: bool
    infinite_shoe: bool


# ── Canonical rule sets ──────────────────────────────────────────────────────

BENCHMARK = Rules(
    variant=RuleVariant.VEGAS_S17,
    blackjack_payout=1.5,
    dealer_stands_soft17=True,
    dealer_peek=True,
    double_allowed="any2",
    double_after_split=True,
    resplit_max_hands=4,
    split_aces_one_card=True,
    surrender_allowed=False,
    insurance_allowed=False,
    infinite_shoe=True,
)

H17_RULES = Rules(
    variant=RuleVariant.VEGAS_H17,
    blackjack_payout=1.5,
    dealer_stands_soft17=False,
    dealer_peek=True,
    double_allowed="any2",
    double_after_split=True,
    resplit_max_hands=4,
    split_aces_one_card=True,
    surrender_allowed=False,
    insurance_allowed=False,
    infinite_shoe=True,
)

S17_SURRENDER = Rules(
    variant=RuleVariant.S17_SURRENDER,
    blackjack_payout=1.5,
    dealer_stands_soft17=True,
    dealer_peek=True,
    double_allowed="any2",
    double_after_split=True,
    resplit_max_hands=4,
    split_aces_one_card=True,
    surrender_allowed=True,
    insurance_allowed=False,
    infinite_shoe=True,
)

S17_NO_DAS = Rules(
    variant=RuleVariant.S17_NO_DAS,
    blackjack_payout=1.5,
    dealer_stands_soft17=True,
    dealer_peek=True,
    double_allowed="any2",
    double_after_split=False,
    resplit_max_hands=4,
    split_aces_one_card=True,
    surrender_allowed=False,
    insurance_allowed=False,
    infinite_shoe=True,
)

ALL_VARIANTS: dict[str, Rules] = {
    "VEGAS_S17":     BENCHMARK,
    "VEGAS_H17":     H17_RULES,
    "S17_SURRENDER": S17_SURRENDER,
    "S17_NO_DAS":    S17_NO_DAS,
}

# ── Action encoding ──────────────────────────────────────────────────────────

ACTION_STAND     = 0
ACTION_HIT       = 1
ACTION_DOUBLE    = 2
ACTION_SPLIT     = 3
ACTION_SURRENDER = 4
NUM_ACTIONS      = 5

ACTION_NAMES = {
    ACTION_STAND:     "Stand",
    ACTION_HIT:       "Hit",
    ACTION_DOUBLE:    "Double",
    ACTION_SPLIT:     "Split",
    ACTION_SURRENDER: "Surrender",
}
ACTION_SHORT = {
    ACTION_STAND:     "S",
    ACTION_HIT:       "H",
    ACTION_DOUBLE:    "D",
    ACTION_SPLIT:     "P",
    ACTION_SURRENDER: "R",
}
