# Gambler
import math, sys
from general import MDP

class MDP_Gambler(MDP):

    def __init__(self, ph):
        self.ph = ph
        self.goal = 16
        states = list(range(self.goal + 1))
        actions = list(range(1, self.goal))
        super().__init__(states, actions, 1.0)

    def dyn(self, s, a):
        if s == 0: return []
        if s == self.goal: return []
        if a > s: return []
        if s + a > self.goal: return []
        return [(self.ph, s + a, 1 if s + a == self.goal else 0),
                (1 - self.ph, s - a, 0)]

gambler = MDP_Gambler(0.4)
(pol, vfn) = gambler.value_iter(thr=0.000001)
print(pol)
eps = 0.00001
for i in range(1, gambler.goal):
    bestA = []
    for a in gambler.actions:
        lst = gambler.dyn(i, a)
        if not lst: continue
        val = sum([p * (rew + vfn[newS]) for (p, newS, rew) in lst])
        if val >= vfn[i] - eps:
            bestA.append(a)
    print(i, vfn[i], bestA)
    
