#!/usr/bin/env python3

import sys
from abc import ABC, abstractmethod

class MDP(ABC):
    def __init__(self, states, actions, gamma, actNbr=None):

        # states: any list
        self.states = states

        # actions: any list
        self.actions = actions

        self.gamma = gamma

        self.actNbr = actNbr

    # dyn(state, action) returns a list of
    # (probability, newState, reward).
    # When a is not allowed in s, then the None should be
    # returned.  Thus, for a terminating state st, self.dyn(st, a)
    # should be None for any action a.
    @abstractmethod
    def dyn(self, s, a):
        pass

    # 
    # pol: policy.  Dictionary.  Key Set is the list of states.
    #   The value for a key s is a list of (probablity, action).
    #   For a terminating state, the value must be the empty list.
    def policy_eval(self, pol, v=None, thr=0.01, rep=None, intRes=0):
        if v is None:
            v = {s:0.0 for s in self.states}
        ir = [(None, v)] if intRes else None
        rcnt = 0
        while True:
            newV = {s : sum([p * p1 * (r + self.gamma * v[s1])
                         for (p, a) in pol[s]
                         for (p1, s1, r) in self.dyn(s, a)])
                    for s in self.states}
            delta = max([abs(newV[s] - v[s]) for s in self.states])
            # print(f'delta = {delta}', file=sys.stderr, flush=True)
            if intRes:
                ir.append((delta, newV))
            if rep:
                rcnt += 1
                if rcnt == rep: break
            if delta < thr:  break
            v = newV
        return (newV, ir)

    def polValFromV(self, v, pol_for_nbr=None):
        def op_s(s):
            best = None
            acts = self.actions \
                if self.actNbr is None \
                   or pol_for_nbr is None \
                   or not pol_for_nbr[s] \
                else self.actNbr[pol_for_nbr[s][0][1]]
            for a in acts:
                lst = self.dyn(s, a)
                if not lst: continue
                if v is None:
                    return (a, 0.0)
                val = sum([p * (r + self.gamma * v[s])
                           for (p, s, r) in lst])
                if best is None or val > best:
                    (bestA, best) = (a, val)
            if best is None:  # terminating state
                return (None, 0.0)
            else:
                return (bestA, best)

        rvA = {}
        rvV = {}
        for s in self.states:
            (xa, xv) = op_s(s)
            rvA[s] = [] if xa is None else [(1.0, xa)]
            rvV[s] = xv
        return (rvA, rvV)

    def policy_iter(self, pol=None, v=None, thr=0.01, intRes=0):
        if v is None:
            v = {s:0.0 for s in self.states}
        if pol is None:
            pol, v = self.polValFromV(v)
        ir = [(pol, v, [])] if intRes else None
        while True:
            v, ir2 = self.policy_eval(pol, v, thr, intRes=(intRes==2))
            new_pol, v = self.polValFromV(v)
            if intRes:
                ir.append((new_pol, v, ir2))
            if all([pol[s] == new_pol[s] for s in v]): break
            pol = new_pol
        return (pol, v, ir)

    def value_iter(self, v = None, thr = 0.01, intRes=0, nbrOnly=False):
        if v is None:
            v = {s:0.0 for s in self.states}
        ir = []
        pol = None
        while True:
            pol_for_nbr = None if (not nbrOnly) or (pol is None) else pol
            pol, newV = self.polValFromV(v, pol_for_nbr=pol_for_nbr)
            delta = max([abs(newV[s] - v[s]) for s in self.states])
            # print(f'delta = {delta}', file=sys.stderr, flush=True)
            if intRes:
                ir.append(delta)
            if delta < thr:  break
            v = newV
        return (pol, newV, ir)

    def value_iter_mix(self, rep=1, v=None, thr=0.01, intRes=0):
        if v is None:
            v = {s:0.0 for s in self.states}
        ir = []
        while True:
            pol, newV = self.polValFromV(v)
            delta = max([abs(newV[s] - v[s]) for s in self.states])
            # print(f'delta = {delta}', file=sys.stderr, flush=True)
            if intRes:
                ir.append(delta)
            if delta < thr:  break
            if rep > 1:
                v, _ = self.policy_eval(pol, v, thr=thr, rep=rep-1)
            else:
                v = newV
        return (pol, newV, ir)

    def show_policy(self, pol): str(pol)
    def showValFn(self, vf) : str(vf)
