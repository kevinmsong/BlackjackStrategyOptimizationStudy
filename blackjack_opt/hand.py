"""
hand.py — Hand dataclass tracking all per-hand state.

The soft/hard total algorithm is the most error-prone piece and is called
in every hot path, so it is kept simple and well-tested.
"""
from __future__ import annotations
from dataclasses import dataclass, field


def compute_total(card_values: list[int]) -> tuple[int, bool]:
    """
    Compute (total, is_soft) from a list of card integer values.
    Aces start at 11; reduce by 10 for each Ace while total > 21.
    Returns (best_total, is_soft) where is_soft means at least one
    Ace is still counted as 11.
    """
    total = 0
    n_aces_as_11 = 0
    for v in card_values:
        total += v
        if v == 11:
            n_aces_as_11 += 1
    while total > 21 and n_aces_as_11 > 0:
        total -= 10
        n_aces_as_11 -= 1
    return total, n_aces_as_11 > 0


@dataclass
class Hand:
    cards: list[int]      # card integer values in order dealt
    total: int            # current best total (soft counts Ace as 11)
    is_soft: bool         # True if an Ace is counted as 11
    is_busted: bool       # total > 21
    is_blackjack: bool    # 2 cards, total 21, not from a split
    pair_rank: int | None # shared value if exactly 2 equal-value cards
    from_split: bool      # hand created by splitting
    split_aces: bool      # created by splitting aces → one card only, no actions
    doubled: bool         # player has doubled → no more hits
    resolved: bool        # hand needs no more player decisions
    n_cards: int          # cached len(cards)

    @classmethod
    def from_cards(
        cls,
        values: list[int],
        from_split: bool = False,
        split_aces: bool = False,
    ) -> "Hand":
        total, is_soft = compute_total(values)
        n = len(values)
        is_busted = total > 21
        is_blackjack = (n == 2) and (total == 21) and (not from_split)
        pair_rank = values[0] if (n == 2 and values[0] == values[1]) else None
        # Split-ace hands with one card dealt are immediately resolved
        resolved = is_busted or is_blackjack or split_aces
        return cls(
            cards=list(values),
            total=total,
            is_soft=is_soft,
            is_busted=is_busted,
            is_blackjack=is_blackjack,
            pair_rank=pair_rank,
            from_split=from_split,
            split_aces=split_aces,
            doubled=False,
            resolved=resolved,
            n_cards=n,
        )

    def add_card(self, value: int) -> None:
        """Update in-place after drawing one more card."""
        self.cards.append(value)
        self.n_cards += 1
        self.total, self.is_soft = compute_total(self.cards)
        self.is_busted = self.total > 21
        # Pair only valid on exactly 2 cards
        if self.n_cards != 2:
            self.pair_rank = None
        # Blackjack can only arise on initial 2-card non-split deal
        self.is_blackjack = (
            self.n_cards == 2
            and self.total == 21
            and not self.from_split
        )
        if self.is_busted:
            self.resolved = True

    def clone(self) -> "Hand":
        return Hand(
            cards=list(self.cards),
            total=self.total,
            is_soft=self.is_soft,
            is_busted=self.is_busted,
            is_blackjack=self.is_blackjack,
            pair_rank=self.pair_rank,
            from_split=self.from_split,
            split_aces=self.split_aces,
            doubled=self.doubled,
            resolved=self.resolved,
            n_cards=self.n_cards,
        )
