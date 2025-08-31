# 480-Assign2
CSC 480 - Poker Win Estimator with MCTS:

A preflop estimator to find the chances of a hand winning a Texas Hold'em game. This uses MCTS to approximate the probability of winning at showdown against a random opponent.

Instructions:
- (have venv running if needed)
- Simple example command:
  ''' python estimator.py --hand "Ah Kh" '''
- For different exploration constant, seed, and # of sims:
  ''' python estimator.py --hand "7c 7d" --sims 25000 --c 1.2 --seed 480 '''
- Limit the maximum number of child nodes expansion:
  ''' python estimator.py --hand "Qs Js" --children 500 '''

Acceptable hand formats: "Ah Kh", "Ah,Kh", or "AhKh"
Default settings: 20k sims, c = sqrt(2), children - 1000, seed - random
