
# estimator.py
# Minimal single-file MCTS estimator for 2-player, no-betting Texas Hold'em.
# Simple comments only.

from __future__ import annotations
import argparse, math, random
from dataclasses import dataclass, field
from collections import Counter
from itertools import combinations
from typing import Iterable, Tuple, List, Optional, Set

# -------------------- Cards helpers --------------------

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

def card_str_to_int(cs: str) -> int:
    # Convert like 'Ah' to 0..51 (rank-major).
    r, s = cs[0].upper(), cs[1].lower()
    ri = RANKS.index(r)
    si = SUITS.index(s)
    return ri * 4 + si

def card_int_to_str(ci: int) -> str:
    # Convert 0..51 back to 'Ah' form.
    ri, si = divmod(ci, 4)
    return RANKS[ri] + SUITS[si]

def parse_two_cards(hand: str) -> Tuple[int, int]:
    # Accepts 'Ah Kh', 'Ah,Kh', or 'AhKh'; returns sorted ints.
    s = hand.replace(",", " ").strip()
    toks = [t for t in s.split() if t]
    if len(toks) == 1 and len(toks[0]) == 4:
        toks = [toks[0][:2], toks[0][2:]]
    if len(toks) != 2:
        raise ValueError(f"Expected two cards like 'Ah Kh', got: {hand!r}")
    a, b = card_str_to_int(toks[0]), card_str_to_int(toks[1])
    if a == b:
        raise ValueError("Duplicate hole cards.")
    return tuple(sorted((a, b)))

def remaining_deck(exclude: Iterable[int]) -> List[int]:
    # Return remaining cards after excluding.
    ex = set(exclude)
    return [i for i in range(52) if i not in ex]

def sample_combos(deck: Iterable[int], combo_size: int, k: int, rng: random.Random) -> List[Tuple[int, ...]]:
    # Sample up to k unique unordered combinations from deck.
    deck_list = list(deck)
    if len(deck_list) < combo_size:
        return []
    seen = set()
    tries = 0
    max_tries = k * 40
    while len(seen) < k and tries < max_tries:
        picks = tuple(sorted(rng.sample(deck_list, combo_size)))
        seen.add(picks)
        tries += 1
    return list(seen)

# -------------------- Hand evaluation --------------------
# Score 5-card hands and pick best 5 out of 7.

# Category ranks (high first):
# 8: Straight Flush, 7: Four of a Kind, 6: Full House, 5: Flush,
# 4: Straight, 3: Trips, 2: Two Pair, 1: One Pair, 0: High Card
def score_5(cards: Iterable[int]) -> Tuple[int, Tuple[int, ...]]:
    ranks = sorted(((c // 4) + 2 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]

    suit_counts = Counter(suits)
    flush_suit = next((s for s, cnt in suit_counts.items() if cnt == 5), None)

    distinct = sorted(set(ranks), reverse=True)
    if 14 in distinct:
        distinct.append(1)  # A-5 straight support
    straight_high = 0
    for i in range(len(distinct) - 4):
        window = distinct[i:i+5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            straight_high = window[0]
            break
    is_straight = straight_high > 0

    if flush_suit is not None and is_straight:
        franks = sorted(((c // 4) + 2 for c in cards if (c % 4) == flush_suit), reverse=True)
        frset = sorted(set(franks), reverse=True)
        if 14 in frset:
            frset.append(1)
        for i in range(len(frset) - 4):
            window = frset[i:i+5]
            if window[0] - window[4] == 4 and len(set(window)) == 5:
                return (8, (window[0],))

    rc = Counter(ranks)
    counts = sorted(((cnt, r) for r, cnt in rc.items()), reverse=True)

    if counts[0][0] == 4:
        four = counts[0][1]
        kicker = max(r for r in ranks if r != four)
        return (7, (four, kicker))

    if counts[0][0] == 3 and any(cnt >= 2 for cnt, _ in counts[1:]):
        trips = counts[0][1]
        pair = next(r for cnt, r in counts[1:] if cnt >= 2)
        return (6, (trips, pair))

    if flush_suit is not None:
        franks = sorted(((c // 4) + 2 for c in cards if (c % 4) == flush_suit), reverse=True)
        return (5, tuple(franks))

    if is_straight:
        return (4, (straight_high,))

    if counts[0][0] == 3:
        t = counts[0][1]
        kickers = [r for r in ranks if r != t][:2]
        return (3, (t, *kickers))

    if counts[0][0] == 2 and counts[1][0] == 2:
        p1, p2 = sorted([counts[0][1], counts[1][1]], reverse=True)
        kicker = max(r for r in ranks if r != p1 and r != p2)
        return (2, (p1, p2, kicker))

    if counts[0][0] == 2:
        p = counts[0][1]
        kickers = [r for r in ranks if r != p][:3]
        return (1, (p, *kickers))

    return (0, tuple(ranks))

def score_best_of_7(cards7: Iterable[int]) -> Tuple[int, Tuple[int, ...]]:
    best = None
    for comb in combinations(cards7, 5):
        sc = score_5(comb)
        if best is None or sc > best:
            best = sc
    return best

# -------------------- MCTS --------------------

@dataclass
class Node:
    hero: Tuple[int, int]
    opp: Optional[Tuple[int, int]] = None
    flop: Optional[Tuple[int, int, int]] = None
    turn: Optional[int] = None
    river: Optional[int] = None

    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)
    untried: List[Tuple[int, ...]] = field(default_factory=list)
    wins: float = 0.0
    visits: int = 0

    max_children: int = 1000
    rng: random.Random = field(default_factory=random.Random)

    def stage(self) -> int:
        # 0=opp, 1=flop, 2=turn, 3=river, 4=terminal
        if self.opp is None: return 0
        if self.flop is None: return 1
        if self.turn is None: return 2
        if self.river is None: return 3
        return 4

    def used_cards(self) -> Set[int]:
        # Collect cards already used
        used = set(self.hero)
        if self.opp: used.update(self.opp)
        if self.flop: used.update(self.flop)
        if self.turn is not None: used.add(self.turn)
        if self.river is not None: used.add(self.river)
        return used

    def is_terminal(self) -> bool:
        return self.stage() == 4

    def _gen_children_pool(self):
        # Sample up to max_children legal next moves
        deck = remaining_deck(self.used_cards())
        st = self.stage()
        if st == 0:
            self.untried = sample_combos(deck, 2, self.max_children, self.rng)
        elif st == 1:
            self.untried = sample_combos(deck, 3, self.max_children, self.rng)
        elif st == 2:
            self.untried = [(c,) for c in self.rng.sample(deck, min(self.max_children, len(deck)))]
        elif st == 3:
            self.untried = [(c,) for c in self.rng.sample(deck, min(self.max_children, len(deck)))]
        else:
            self.untried = []

    def expand(self) -> Optional['Node']:
        # Expand one child
        if not self.untried:
            self._gen_children_pool()
        if not self.untried:
            return None
        move = self.untried.pop()
        st = self.stage()
        if st == 0:
            child = Node(hero=self.hero, opp=(move[0], move[1]), parent=self, rng=self.rng, max_children=self.max_children)
        elif st == 1:
            child = Node(hero=self.hero, opp=self.opp, flop=(move[0], move[1], move[2]), parent=self, rng=self.rng, max_children=self.max_children)
        elif st == 2:
            child = Node(hero=self.hero, opp=self.opp, flop=self.flop, turn=move[0], parent=self, rng=self.rng, max_children=self.max_children)
        elif st == 3:
            child = Node(hero=self.hero, opp=self.opp, flop=self.flop, turn=self.turn, river=move[0], parent=self, rng=self.rng, max_children=self.max_children)
        else:
            return None
        self.children.append(child)
        return child

    def ucb1(self, c: float) -> float:
        # UCB1 selection value
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float) -> 'Node':
        # Pick child with max UCB1
        return max(self.children, key=lambda ch: ch.ucb1(c))

    def select(self, c: float) -> 'Node':
        # Tree policy: descend by UCB1, expand when possible
        node = self
        while not node.is_terminal():
            if node.untried:
                return node.expand()
            if not node.children:
                exp = node.expand()
                if exp is not None:
                    return exp
                break
            node = node.best_child(c)
        return node

    def rollout(self) -> float:
        # Play out random cards to the river, then evaluate
        rng = self.rng
        used = self.used_cards()
        deck = remaining_deck(used)

        opp = self.opp or tuple(sorted(rng.sample(deck, 2)))
        for c in opp:
            if c in deck: deck.remove(c)

        flop = self.flop or tuple(sorted(rng.sample(deck, 3)))
        for c in flop:
            if c in deck: deck.remove(c)

        turn = self.turn or rng.choice(deck)
        if turn in deck: deck.remove(turn)

        river = self.river or rng.choice(deck)

        hero7 = list(self.hero) + list(flop) + [turn, river]
        opp7  = list(opp)       + list(flop) + [turn, river]

        h = score_best_of_7(hero7)
        o = score_best_of_7(opp7)
        if h > o: return 1.0
        if h < o: return 0.0
        return 0.5

    def backprop(self, result: float):
        # Update path stats
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

class MCTS:
    # Run MCTS from the root state
    def __init__(self, hero: Tuple[int, int], c: float, max_children: int, seed: int | None):
        self.rng = random.Random(seed)
        self.root = Node(hero=hero, max_children=max_children, rng=self.rng)
        self.c = c

    def run(self, simulations: int) -> Tuple[float, int]:
        # Do N simulations and return (win_prob, sims_done)
        for _ in range(simulations):
            leaf = self.root.select(self.c)
            result = leaf.rollout()
            leaf.backprop(result)
        return (self.root.wins / self.root.visits, self.root.visits)

# -------------------- CLI --------------------

def main():
    p = argparse.ArgumentParser(description="Single-file 2-player, no-betting Hold'em win estimator using MCTS.")
    p.add_argument("--hand", required=True, help="Your two hole cards, e.g., 'Ah Kh' or 'AhKh'")
    p.add_argument("--sims", type=int, default=20000, help="Number of simulations (default: 20000)")
    p.add_argument("--c", type=float, default=math.sqrt(2), help="UCB1 exploration constant (default: sqrt(2))")
    p.add_argument("--children", type=int, default=1000, help="Max children per node (default: 1000)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = p.parse_args()

    hero = parse_two_cards(args.hand)
    mcts = MCTS(hero, c=args.c, max_children=args.children, seed=args.seed)
    est, sims_done = mcts.run(args.sims)

    a, b = hero
    print("=== Minimal Poker Win Estimator ===")
    print(f"Hand: {card_int_to_str(a)} {card_int_to_str(b)}")
    print(f"Simulations: {sims_done}")
    print(f"Estimated win probability: {est:.4f}")

if __name__ == "__main__":
    main()
