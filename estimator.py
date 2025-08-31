import argparse, math, random
from collections import Counter
from itertools import combinations

# --- Cards ---
RANKS="23456789TJQKA"; SUITS="cdhs"

def c2i(cs):
    return RANKS.index(cs[0].upper())*4 + SUITS.index(cs[1].lower())

def i2c(i):
    r,s=divmod(i,4); return RANKS[r]+SUITS[s]

def parse_hand(txt):
    s=txt.replace(","," ").strip(); t=[x for x in s.split() if x]
    if len(t)==1 and len(t[0])==4: t=[t[0][:2],t[0][2:]]
    if len(t)!=2: raise ValueError("Use 'Ah Kh' or 'AhKh'")
    a,b=c2i(t[0]),c2i(t[1])
    if a==b: raise ValueError("Duplicate cards")
    return (a,b) if a<b else (b,a)

def rest(ex):
    e=set(ex); return [i for i in range(52) if i not in e]

def sample(deck, size, k, rng):
    if len(deck)<size: return []
    seen, out, tries, cap=set(),[],0,k*40
    d=list(deck)
    while len(out)<k and tries<cap:
        p=tuple(sorted(rng.sample(d,size)))
        if p not in seen: seen.add(p); out.append(p)
        tries+=1
    return out

# --- Hand eval (best 5 of 7) ---
# Category: 8 SF, 7 Quads, 6 Full, 5 Flush, 4 Straight, 3 Trips, 2 TwoPair, 1 Pair, 0 High
def score5(cs):
    rs=sorted(((c//4)+2 for c in cs),reverse=True)
    ss=[c%4 for c in cs]
    fs=None
    for s,n in Counter(ss).items():
        if n==5: fs=s; break
    d=sorted(set(rs),reverse=True)
    if 14 in d: d.append(1)
    sh=0
    for i in range(len(d)-4):
        w=d[i:i+5]
        if w[0]-w[4]==4 and len(set(w))==5: sh=w[0]; break
    if fs is not None and sh:
        fr=sorted(((c//4)+2 for c in cs if c%4==fs),reverse=True)
        u=sorted(set(fr),reverse=True)
        if 14 in u: u.append(1)
        for i in range(len(u)-4):
            w=u[i:i+5]
            if w[0]-w[4]==4 and len(set(w))==5: return (8,(w[0],))
    rc=Counter(rs)
    cnt=sorted(((v,k) for k,v in rc.items()),reverse=True)
    if cnt[0][0]==4:
        q=cnt[0][1]; k=max(r for r in rs if r!=q); return (7,(q,k))
    if cnt[0][0]==3:
        pr=None
        for v,k in cnt[1:]:
            if v>=2: pr=k; break
        if pr is not None: return (6,(cnt[0][1],pr))
    if fs is not None:
        fr=sorted(((c//4)+2 for c in cs if c%4==fs),reverse=True); return (5,tuple(fr))
    if sh: return (4,(sh,))
    if cnt[0][0]==3:
        t=cnt[0][1]; ks=[r for r in rs if r!=t][:2]; return (3,(t,*ks))
    if cnt[0][0]==2 and cnt[1][0]==2:
        p1,p2=sorted([cnt[0][1],cnt[1][1]],reverse=True)
        k=max(r for r in rs if r!=p1 and r!=p2); return (2,(p1,p2,k))
    if cnt[0][0]==2:
        p=cnt[0][1]; ks=[r for r in rs if r!=p][:3]; return (1,(p,*ks))
    return (0,tuple(rs))

def best7(cs7):
    best=None
    for comb in combinations(cs7,5):
        sc=score5(comb)
        if best is None or sc>best: best=sc
    return best

# --- MCTS ---
class Node:
    def __init__(self, hero, parent=None, rng=None, maxch=1000, opp=None, flop=None, turn=None, river=None):
        self.hero=hero; self.parent=parent; self.rng=rng or random.Random()
        self.maxch=maxch; self.opp=opp; self.flop=flop; self.turn=turn; self.river=river
        self.children=[]; self.untried=[]; self.wins=0.0; self.visits=0

    def stage(self):
        if self.opp is None: return 0
        if self.flop is None: return 1
        if self.turn is None: return 2
        if self.river is None: return 3
        return 4

    def used(self):
        u=set(self.hero)
        if self.opp: u.update(self.opp)
        if self.flop: u.update(self.flop)
        if self.turn is not None: u.add(self.turn)
        if self.river is not None: u.add(self.river)
        return u

    def terminal(self): return self.stage()==4

    def gen(self):
        d=rest(self.used()); s=self.stage()
        if s==0: self.untried=sample(d,2,self.maxch,self.rng)
        elif s==1: self.untried=sample(d,3,self.maxch,self.rng)
        elif s==2: self.untried=[(c,) for c in self.rng.sample(d,min(self.maxch,len(d)))]
        elif s==3: self.untried=[(c,) for c in self.rng.sample(d,min(self.maxch,len(d)))]

    def expand(self):
        if not self.untried: self.gen()
        if not self.untried: return None
        m=self.untried.pop(); s=self.stage()
        if   s==0: ch=Node(self.hero,self,self.rng,self.maxch,opp=(m[0],m[1]),flop=self.flop,turn=self.turn,river=self.river)
        elif s==1: ch=Node(self.hero,self,self.rng,self.maxch,opp=self.opp,flop=(m[0],m[1],m[2]),turn=self.turn,river=self.river)
        elif s==2: ch=Node(self.hero,self,self.rng,self.maxch,opp=self.opp,flop=self.flop,turn=m[0],river=self.river)
        elif s==3: ch=Node(self.hero,self,self.rng,self.maxch,opp=self.opp,flop=self.flop,turn=self.turn,river=m[0])
        else: return None
        self.children.append(ch); return ch

    def ucb1(self,c):
        return float('inf') if self.visits==0 else (self.wins/self.visits)+c*math.sqrt(math.log(self.parent.visits)/self.visits)

    def best_child(self,c):
        b=None; bv=-1e100
        for ch in self.children:
            v=ch.ucb1(c)
            if v>bv: bv=v; b=ch
        return b

    def select(self,c):
        n=self
        while not n.terminal():
            if n.untried: return n.expand()
            if not n.children:
                e=n.expand()
                if e is not None: return e
                break
            n=n.best_child(c)
        return n

    def rollout(self):
        d=rest(self.used()); r=self.rng
        opp=self.opp or tuple(sorted(r.sample(d,2))); [d.remove(x) for x in opp if x in d]
        flp=self.flop or tuple(sorted(r.sample(d,3))); [d.remove(x) for x in flp if x in d]
        trn=self.turn or r.choice(d);  d.remove(trn) if trn in d else None
        rvr=self.river or r.choice(d)
        h=best7(list(self.hero)+list(flp)+[trn,rvr])
        o=best7(list(opp)+list(flp)+[trn,rvr])
        return 1.0 if h>o else 0.0 if h<o else 0.5

    def backprop(self,res):
        n=self
        while n is not None:
            n.visits+=1; n.wins+=res; n=n.parent

class MCTS:
    def __init__(self, hero, c, maxch, seed):
        self.root=Node(hero, rng=random.Random(seed), maxch=maxch); self.c=c
    def run(self, sims):
        for _ in range(sims):
            leaf=self.root.select(self.c); leaf.backprop(leaf.rollout())
        return self.root.wins/self.root.visits

# --- CLI ---
def main():
    ap=argparse.ArgumentParser(description="Preflop MCTS estimator (2-player)")
    ap.add_argument("--hand", required=True, help="Hole cards: 'Ah Kh' / 'Ah,Kh' / 'AhKh'")
    ap.add_argument("--sims", type=int, default=20000, help="MCTS simulations (default 20000)")
    ap.add_argument("--c", type=float, default=math.sqrt(2), help="UCB1 exploration constant (default sqrt(2))")
    ap.add_argument("--children", type=int, default=1000, help="Max children per node (default 1000)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args=ap.parse_args()

    hero=parse_hand(args.hand)
    est=MCTS(hero, c=args.c, maxch=args.children, seed=args.seed).run(args.sims)
    a,b=hero
    print("=== Preflop Win Probability (2-player) ===")
    print(f"Hand: {i2c(a)} {i2c(b)}")
    print(f"Simulations: {args.sims}")
    print(f"Estimate: {est:.4f}")

if __name__=="__main__":
    main()