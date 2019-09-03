#!/usr/bin/env python3

# Jack's Car Rental
import math, sys, random
from general import MDP

class CRStation:
    def __init__(self, cap, expRent, expRet):
        self.cap = cap; self.expRent = expRent; self.expRet = expRet

class MDP_CarRental(MDP):

    def nbr_act(self, a):
        if a == self.maxTrans:    return [a-1, a]
        elif a == -self.maxTrans: return [a, a+1]
        else:                     return [a-1, a, a+1]

    def __init__(self):
        self.maxTrans = 5
        self.rewRent = 10
        self.costTrans = 2
        self.stn = [CRStation(20, 3, 3), CRStation(20, 4, 2)]
        # self.stn = [CRStation(20, 8, 8), CRStation(20, 12, 9)]
        # self.stn = [CRStation(5, 3, 3), CRStation(5, 4, 2)]

        states = [(x,y) for x in range(self.stn[0].cap + 1)
                        for y in range(self.stn[1].cap + 1)]
        actions = list(range(-self.maxTrans, self.maxTrans + 1))
        actNbr = { a: self.nbr_act(a) for a in actions }
        super().__init__(states, actions, 0.9, actNbr=actNbr)

        self.poisson = {}
        self.poissonA = {}
        for i in [0,1]:
            for lam in [self.stn[i].expRent, self.stn[i].expRet]:
                self.calcPoisson(lam, self.stn[i].cap)
        self.prep_dyn()

    def calcPoisson(self, lam, maxN):
        if lam in self.poisson and len(self.poisson[lam]) >= maxN + 1:
            return
        e = math.exp(-lam)
        last = e
        lastA = 1.0
        self.poisson[lam] = [last]
        self.poissonA[lam] = [lastA]
        for n in range(1, maxN+1):
            lastA -= last
            last *= lam / n
            self.poisson[lam].append(last)
            self.poissonA[lam].append(lastA)

    # The probability and combined reward for transfering from
    # m to k; i.e. car number was m in the beginning of the day
    # and k at the end of the day.  There should exist number w
    # such that w cars were rent and k-(m-w) were returned.
    # Note that in the cases of w == m and k == cap, probability
    # should be accumulated.
    def rentRet(self, i, m, k):
        pSum = 0.0
        rSum = 0.0
        pWArr = self.poissonA if k == self.stn[i].cap else self.poisson
        for w in range(m+1):
            if m - w > k: continue
            pVArr = self.poissonA if w == m else self.poisson
            pV = pVArr[self.stn[i].expRent][w]
            pW = pWArr[self.stn[i].expRet][k - (m-w)]
            p = pV * pW
            pSum += p
            r = self.rewRent * w
            rSum += p * r
        return (pSum, rSum / pSum)

    def prep_dyn(self):
        self.tblRR = [[[self.rentRet(i, m, k)
                        for k in range(self.stn[i].cap + 1)]
                       for m in range(self.stn[i].cap + 1)]
                      for i in [0,1]]

    def dyn(self, s, a):
        rv = []
        (n0, n1) = s
        (m0, m1) = (n0 + a, n1 - a)
        if m0 < 0 or m1 < 0 \
           or m0 > self.stn[0].cap or m1 > self.stn[1].cap:
            return []
        rewT = -self.costTrans * abs(a)
        for k0 in range(self.stn[0].cap + 1):
            for k1 in range(self.stn[1].cap + 1):
                (p0, r0) = self.tblRR[0][m0][k0]
                (p1, r1) = self.tblRR[1][m1][k1]
                rv.append((p0 * p1, (k0, k1), rewT + r0 + r1))
        return rv
    
    def simulate(self, pol, init0, init1, duration):
        def rand(lam, maxN):
            for j in range(1, maxN+1):
                x = random.random()
                if self.poissonA[lam][j] < x: return (j-1)
            return maxN

        m0, m1 = init0, init1
        revenue = 0
        for i in range(duration):
            reqRent0 = rand(self.stn[0].expRent, self.stn[0].cap)
            reqRent1 = rand(self.stn[1].expRent, self.stn[1].cap)
            reqRet0 = rand(self.stn[0].expRet, self.stn[0].cap)
            reqRet1 = rand(self.stn[0].expRet, self.stn[1].cap)
            rent0 = min(reqRent0, m0)
            rent1 = min(reqRent1, m1)
            m0 = min(m0 - rent0 + reqRet0, self.stn[0].cap)
            m1 = min(m1 - rent1 + reqRet1, self.stn[1].cap)
            a = pol[(m0, m1)][0][1]
            m0 += a
            m1 -= a
            revenue += self.rewRent * (rent0 + rent1) - a * self.costTrans
        return revenue

    def write_policy(self, pol, fp):
        for n0 in range(self.stn[0].cap + 1):
            for n1 in range(self.stn[1].cap + 1):
                print(f'{pol[(n0,n1)][0][1] : 2}', end='', file=fp)
            print(file=fp)
        print(file=fp)

carRental = MDP_CarRental()
pol0 = { s : [(1.0, 0)] for s in carRental.states }

def doPolicyIter():
    (pol, v, ir) = carRental.policy_iter(pol=pol0, thr=0.01, intRes=2)    
    log = 'logCRa.txt'
    with open(log, 'w') as fp:
        for (pol, v, ir2) in ir:
            print('iter for eval: ', len(ir2), file=fp)
            carRental.write_policy(pol, fp)
        print(f'Printed to {log}')

def doSimulation():
    duration = 300
    init0, init1 = 10, 10
    iteration = 100
    polOpt, _, _ = carRental.policy_iter(thr=0.01)
    rev0 = [carRental.simulate(pol0, init0, init1, duration)
            for _ in range(iteration)]
    revOpt = [carRental.simulate(polOpt, init0, init1, duration)
              for _ in range(iteration)]
    print(sum(rev0), rev0)
    print(sum(revOpt), revOpt)
        
def doValueIter():
    pol, v, ir = carRental.value_iter(thr=0.01, intRes=1, nbrOnly=True)
    # pol, v, ir = carRental.value_iter_mix(thr=0.01, rep=20, intRes=1)
    print(ir)

doPolicyIter()
# doSimulation()
# doValueIter()


