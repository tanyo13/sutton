#!/usr/bin/env python3

import sys
from general import MDP

class MDP_ex41(MDP):
    dirs = { 'U': (-1,0), 'D': (1,0), 'R': (0,1), 'L': (0,-1) }
    
    def dyn(self, s, a):

        def nxt(s, a):
            dy, dx = MDP_ex41.dirs[a]
            y, x = s // 4, s % 4
            st = 4 * min(3, max(0, y + dy)) + min(3, max(0, x + dx))
            return 0 if st == 15 else st

        if s == 0: return None
        return [(1, nxt(s, a), -1)]

    def __init__(self):
        super().__init__(list(range(15)), 'URDL', 1.0)

    def showValFn(self, v):
        def sub(i):
            if i == 0 or i == 15: return f'{0: 7.2f}'
            return f'{v[i]: 7.2f}'
        return '\n'.join([''.join([sub(y*4 + x) for x in range(4)])
                          for y in range(4)])

    def saveIntRes(self, ir, fpath):
        with open(fpath, 'w') as fp:
            for (i, (delta, v)) in enumerate(ir):
                print(f'iter = {i:4}, delta = {delta}', file=fp)
                print(mdp_e41.showValFn(v), file=fp)
                print(file=fp)
        print(f'Log was printed into {fpath}.', file=sys.stderr)

mdp_e41 = MDP_ex41()

pol_e41 = { i : [ (0.25, a) for a in mdp_e41.actions ]
            for i in range(1, 15) }
pol_e41[0] = []

def exc_41():
    v, ir = mdp_e41.policy_eval(pol_e41, intRes=1, thr=0.0001)
    mdp_e41.saveIntRes(ir, 'log41.txt')

def eval_opt_policy():
    v0, _ = mdp_e41.policy_eval(pol_e41, thr=0.1)
    pol_opt, _ = mdp_e41.polValFromV(v0)
    v, ir = mdp_e41.policy_eval(pol_opt, intRes=1, thr=0.0001)
    mdp_e41.saveIntRes(ir, 'log41a.txt')

# exc_41()
eval_opt_policy()

