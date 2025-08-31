import argparse
import math
import random
from collections import Counter
from itertools import combinations

# ----- init Cards ----

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

def card_str_to_int(cs):
    # Convert 'Ah' -> integer 0..51 (rank-major)
    r, s = cs[0].upper(), cs[1].lower()
    return RANKS.index(r) * 4 + SUITS.index(s)

def card_int_to_str(ci):
    # Convert integer 0..51 back to string like 'Ah'
    ri, si = divmod(ci, 4)
    return RANKS[ri] + SUITS[si]

def parse_two_cards(text):
    # takes 'Ah Kh', 'Ah,Kh', or 'AhKh'; returns sorted ints (a, b)
    s = text.replace(",", " ").strip()
    toks = [t for t in s.split() if t]
    if len(toks) == 1 and len(toks[0]) == 4:
        toks = [toks[0][:2], toks[0][2:]]
    if len(toks) != 2:
        raise ValueError("Provide two cards (e.g., 'Ah Kh' or 'AsKd')")
    a, b = card_str_to_int(toks[0]), card_str_to_int(toks[1])
    if a == b:
        raise ValueError("Duplicate hole cards")
    if a > b:
        a, b = b, a
    return a, b

def parse_hole_cards_4char(compact):
    # takes 'AsKh' or 'TcTd' (exactly 4 chars)
    if len(compact) != 4:
        raise ValueError("Use 4-char format like AsKh or TcTd")
    a, b = card_str_to_int(compact[:2]), card_str_to_int(compact[2:])
    if a == b:
        raise ValueError("Duplicate hole cards")
    if a > b:
        a, b = b, a
    return a, b

def remaining_deck(exclude):
    # Return deck without excluded cards
    ex = set(exclude)
    return [i for i in range(52) if i not in ex]

def sample_combos(deck, size, k, rng):
    # Sample up to k unique unordered combinations from deck
    deck_list = list(deck)
    if len(deck_list) < size:
        return []
    seen = set()
    tries = 0
    cap = k * 40  # simple cap to avoid loops
    while len(seen) < k and tries < cap:
        picks = tuple(sorted(rng.sample(deck_list, size)))
        seen.add(picks)
        tries += 1
    return list(seen)

# ---- Hand eval (best 5 of 7) ----

# Category order (high to low):
# 8: Straight Flush, 7: Four of a Kind, 6: Full House, 5: Flush,
# 4: Straight, 3: Three of a Kind, 2: Two Pair, 1: One Pair, 0: High Card
def score_5(cards):
    # cards: iterable of 5 ints 0..51
    ranks = sorted(((c // 4) + 2 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]

    # flush check
    suit_counts = Counter(suits)
    flush_suit = None
    for s, n in suit_counts.items():
        if n == 5:
            flush_suit = s
            break

    # straight check (with wheel)
    distinct = sorted(set(ranks), reverse=True)
    if 14 in distinct:
        distinct.append(1)  # ace-low
    straight_high = 0
    for i in range(len(distinct) - 4):
        w = distinct[i:i+5]
        if w[0] - w[4] == 4 and len(set(w)) == 5:
            straight_high = w[0]
            break
    is_straight = straight_high > 0

    # straight flush
    if flush_suit is not None and is_straight:
        fr = sorted(((c // 4) + 2 for c in cards if (c % 4) == flush_suit), reverse=True)
        frd = sorted(set(fr), reverse=True)
        if 14 in frd:
            frd.append(1)
        for i in range(len(frd) - 4):
            w = frd[i:i+5]
            if w[0] - w[4] == 4 and len(set(w)) == 5:
                return (8, (w[0],))

    # multiples
    rc = Counter(ranks)
    counts = sorted(((cnt, r) for r, cnt in rc.items()), reverse=True)

    # four of a kind
    if counts[0][0] == 4:
        four = counts[0][1]
        kicker = max(r for r in ranks if r != four)
        return (7, (four, kicker))

    # full house
    if counts[0][0] == 3:
        # check for any pair in the rest
        pair_rank = None
        for cnt, r in counts[1:]:
            if cnt >= 2:
                pair_rank = r
                break
        if pair_rank is not None:
            return (6, (counts[0][1], pair_rank))

    # flush
    if flush_suit is not None:
        fr = sorted(((c // 4) + 2 for c in cards if (c % 4) == flush_suit), reverse=True)
        return (5, tuple(fr))

    # straight
    if is_straight:
        return (4, (straight_high,))

    # three of a kind
    if counts[0][0] == 3:
        t = counts[0][1]
        kickers = [r for r in ranks if r != t][:2]
        return (3, (t, *kickers))

    # two pair
    if counts[0][0] == 2 and counts[1][0] == 2:
        p1, p2 = sorted([counts[0][1], counts[1][1]], reverse=True)
        kicker = max(r for r in ranks if r != p1 and r != p2)
        return (2, (p1, p2, kicker))

    # one pair
    if counts[0][0] == 2:
        p = counts[0][1]
        kickers = [r for r in ranks if r != p][:3]
        return (1, (p, *kickers))

    # high card
    return (0, tuple(ranks))

def score_best_of_7(cards7):
    # pick best 5-card score from 7 cards
    best = None
    for comb in combinations(cards7, 5):
        sc = score_5(comb)
        if best is None or sc > best:
            best = sc
    return best

# ----- begin MCTS ----

class Node:
    # a single state in the (chance) game tree
    def __init__(self, hero, parent=None, rng=None, max_children=1000,
                 opp=None, flop=None, turn=None, river=None):
        self.hero = hero
        self.opp = opp
        self.flop = flop
        self.turn = turn
        self.river = river

        self.parent = parent
        self.children = []
        self.untried = []
        self.wins = 0.0
        self.visits = 0
        self.max_children = max_children
        self.rng = rng if rng is not None else random.Random()

    def stage(self):
        # 0=pick opp, 1=pick flop, 2=pick turn, 3=pick river, 4=terminal
        if self.opp is None:  return 0
        if self.flop is None: return 1
        if self.turn is None: return 2
        if self.river is None: return 3
        return 4

    def used_cards(self):
        # which cards are already fixed
        used = set(self.hero)
        if self.opp:  used.update(self.opp)
        if self.flop: used.update(self.flop)
        if self.turn is not None:  used.add(self.turn)
        if self.river is not None: used.add(self.river)
        return used

    def is_terminal(self):
        return self.stage() == 4

    def _gen_children_pool(self):
        # generate up to max_children legal next moves
        deck = remaining_deck(self.used_cards())
        st = self.stage()
        if st == 0:
            self.untried = sample_combos(deck, 2, self.max_children, self.rng)
        elif st == 1:
            self.untried = sample_combos(deck, 3, self.max_children, self.rng)
        elif st == 2:
            n = min(self.max_children, len(deck))
            self.untried = [(c,) for c in self.rng.sample(deck, n)]
        elif st == 3:
            n = min(self.max_children, len(deck))
            self.untried = [(c,) for c in self.rng.sample(deck, n)]
        else:
            self.untried = []

    def expand(self):
        # take one move from untried and create child
        if not self.untried:
            self._gen_children_pool()
        if not self.untried:
            return None
        move = self.untried.pop()
        st = self.stage()
        if st == 0:
            child = Node(self.hero, parent=self, rng=self.rng, max_children=self.max_children,
                         opp=(move[0], move[1]), flop=self.flop, turn=self.turn, river=self.river)
        elif st == 1:
            child = Node(self.hero, parent=self, rng=self.rng, max_children=self.max_children,
                         opp=self.opp, flop=(move[0], move[1], move[2]), turn=self.turn, river=self.river)
        elif st == 2:
            child = Node(self.hero, parent=self, rng=self.rng, max_children=self.max_children,
                         opp=self.opp, flop=self.flop, turn=move[0], river=self.river)
        elif st == 3:
            child = Node(self.hero, parent=self, rng=self.rng, max_children=self.max_children,
                         opp=self.opp, flop=self.flop, turn=self.turn, river=move[0])
        else:
            return None
        self.children.append(child)
        return child

    def ucb1(self, c):
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c):
        # iterate to avoid lambdas
        best = None
        best_val = -1e100
        for ch in self.children:
            val = ch.ucb1(c)
            if val > best_val:
                best_val = val
                best = ch
        return best

    def select(self, c):
        # selection + expansion
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

    def rollout(self):
        # complete to river randomly and score
        deck = remaining_deck(self.used_cards())
        rng = self.rng

        opp = self.opp or tuple(sorted(rng.sample(deck, 2)))
        for c in opp:
            if c in deck:
                deck.remove(c)

        flop = self.flop or tuple(sorted(rng.sample(deck, 3)))
        for c in flop:
            if c in deck:
                deck.remove(c)

        turn = self.turn or rng.choice(deck)
        if turn in deck:
            deck.remove(turn)

        river = self.river or rng.choice(deck)

        hero7 = list(self.hero) + list(flop) + [turn, river]
        opp7  = list(opp)       + list(flop) + [turn, river]

        h = score_best_of_7(hero7)
        o = score_best_of_7(opp7)
        if h > o:
            return 1.0
        if h < o:
            return 0.0
        return 0.5

    def backprop(self, result):
        # update stats up the path
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

class MCTS:
    def __init__(self, hero, c, max_children, seed=None):
        self.rng = random.Random(seed)
        self.root = Node(hero, rng=self.rng, max_children=max_children)
        self.c = c

    def run(self, sims):
        for _ in range(sims):
            leaf = self.root.select(self.c)
            res = leaf.rollout()
            leaf.backprop(res)
        return self.root.wins / self.root.visits, self.root.visits


def estimate_win_probability(hole, sims=20000, c=math.sqrt(2), children=1000, seed=None):
    m = MCTS(hole, c=c, max_children=children, seed=seed)
    p, _ = m.run(sims)
    return p

def interactive():
    print("=" * 58)
    print("Texas Hold'em MCTS Preflop Win Probability (2-player, no betting)")
    print("=" * 58)
    print("Enter hole cards as 4 chars (AsKh, TcTd). Type 'q' to quit.\n")
    while True:
        s = input("Hole cards> ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            hero = parse_hole_cards_4char(s)
            sims_in = input("Simulations [10000]: ").strip()
            sims = int(sims_in) if sims_in else 10000
            p = estimate_win_probability(hero, sims=sims)
            a, b = hero
            print(f"{card_int_to_str(a)} {card_int_to_str(b)} -> preflop equity: {p:.3f}\n")
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    ap = argparse.ArgumentParser(description="Preflop MCTS estimator (2-player, no betting).")
    ap.add_argument("--hand", help="Hole cards: 'Ah Kh' / 'Ah,Kh' / 'AhKh' or compact 'AsKh'")
    ap.add_argument("--sims", type=int, default=20000, help="MCTS simulations (default 20000)")
    ap.add_argument("--c", type=float, default=math.sqrt(2), help="UCB1 exploration constant (default sqrt(2))")
    ap.add_argument("--children", type=int, default=1000, help="Max children per node (default 1000)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = ap.parse_args()

    if args.interactive or not args.hand:
        interactive()
        return

    # parse --hand in either spaced/comma or 4-char form
    try:
        if len(args.hand.replace(" ", "").replace(",", "")) == 4:
            hero = parse_hole_cards_4char(args.hand.replace(" ", "").replace(",", ""))
        else:
            hero = parse_two_cards(args.hand)
    except Exception as e:
        raise SystemExit(f"Bad --hand value: {e}")

    p = estimate_win_probability(hero, sims=args.sims, c=args.c, children=args.children, seed=args.seed)
    a, b = hero
    print("=== Preflop Win Probability (2-player, no betting) ===")
    print(f"Hand: {card_int_to_str(a)} {card_int_to_str(b)}")
    print(f"Simulations: {args.sims}")
    print(f"Estimate: {p:.4f}")

if __name__ == "__main__":
    main()