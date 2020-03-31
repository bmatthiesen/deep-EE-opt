#!/usr/bin/env python3

# Copyright (C) 2018-2020 Bho Matthiesen, Karl-Ludwig Besser
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and
# Merouane Debbah, "A Globally Optimal Energy-Efficient Power Control Framework
# and its Efficient Implementation in Wireless Interference Networks,"
# submitted to IEEE Transactions on Signal Processing
# 
# License:
# This program is licensed under the GPLv2 license. If you in any way use this
# code for research that results in publications, please cite our original
# article listed above.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import cvxpy as cp

def SCA(h, mu, Pc, Pmax, pt = None, MaxIter = 10000, parm_alpha = 1e-8, parm_beta = 0.01, RelTolFun = 1e-12, RelTolVal = 1e-12):
    if pt is None:
        pt = np.full(h.shape[-1], Pmax)

    def f(p): # verified
        s = h * p
        direct = np.diag(s)
        ifn = 1 + np.sum(s, axis=-1) - direct
        rates = np.log(1+direct/ifn)
        ee = rates / (mu * p + Pc)

        return np.sum(ee)

    def gradr(p): # verified
        s = h * p
        tmp = 1 + np.sum(s, axis=-1) # 1 + sum beta + a
        tmp2 = tmp - np.diag(s)
        fac = np.diag(s) / (tmp * tmp2)

        grad = h.copy()
        grad = -(fac * grad.T).T

        grad[np.diag_indices_from(grad)] = tmp2/(tmp*tmp2) * np.diag(h)

        return grad

    def gradf(p): # verified
        tmp = 1 / (mu * p + Pc)
        gr = gradr(p)
        t1 = np.sum((gr.T * tmp).T, axis=0)

        s = h * p
        direct = np.diag(s)
        ifn = 1 + np.sum(s, axis=-1) - direct
        rates = np.log(1+direct/ifn)
        t2 = mu * rates * tmp**2

        return t1 - t2

    # gradient step parameter
    cnt = 0
    obj = f(pt)
    while True:
        cnt += 1

        # grad r (without main diagonal)
        s = h * pt
        tmp = 1 + np.sum(s, axis=-1) # 1 + sum beta + a
        tmp2 = tmp - np.diag(s)
        fac = np.diag(s) / (tmp * tmp2)

        beta = h.copy()
        beta[np.diag_indices_from(beta)] = 0
        grad = -(fac * beta.T).T

        """
        # test
        g2 = gradr(pt)
        g2[np.diag_indices_from(g2)] = 0
        assert(np.allclose(g2,grad))
        """

        # r tilde constants
        txp = 1.0/(mu * pt + Pc)

        c1 = np.sum(grad * txp, axis=0)
        c2 = -mu * np.log(np.diag(s)/tmp2+1)*txp**2
        c = c1+c2

        d = -c * pt

        # solve inner problem
        pvar = cp.Variable(4)
        obj_nl = cp.log(cp.multiply(np.diag(h)/tmp2, pvar)+1) * txp
        obj_l  = cp.multiply(c, pvar)

        objective = cp.Maximize(cp.sum(obj_nl + obj_l + d))
        constraints = [0 <= pvar, pvar <= Pmax]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except cp.SolverError:
            print('ecos failed')
            try:
                prob.solve(solver = cp.CVXOPT)
            except cp.SolverError:
                print('cvxopt also failed')
                break
        
        # calculate gradient step
        Bpt = pvar.value - pt
        gamma = 1

        old_obj = obj # f(pt)
        old_pt = pt
        while f(pt + gamma * Bpt) < old_obj + parm_alpha * gamma * gradf(pt) @ Bpt:
            gamma *= parm_beta

        pt += gamma * Bpt
        obj = f(pt)

        with np.errstate(divide='ignore'):
            if abs(obj/old_obj - 1) < RelTolFun and np.linalg.norm(pt-old_pt, np.inf) / np.linalg.norm(pt, np.inf) < RelTolVal:
                break
        
        if cnt > MaxIter:
            print('MaxIter')
            break

    return (obj, pt)

def SCA_randinit(num, h, mu, Pc):
    dim = h.shape[-1]

    obj = -np.inf
    for cnt in range(num):
        pt = np.random.rand(dim)
        obj1, popt1 = SCA(h, mu, Pc, pt = pt)

        if obj1 > obj:
            obj = obj1
            popt = popt1

    return (obj, popt)

if __name__ == "__main__":
    import progressbar as pb
    import itertools as it
    import h5py

    dfn = '../../data/results.h5'
    mu = 4
    Pc = 1

    f = h5py.File(dfn, 'a')
    dset = f

    Plin = 10**(np.asarray(dset['input/PdB'][...]/10))

    try:
        obj = dset.create_dataset('SCA', shape = dset['objval'].shape, fillvalue = np.nan, dtype = dset['objval'].dtype)
        popt = dset.create_dataset('SCA_xopt', shape = dset['objval'].shape + (4,), fillvalue = np.nan, dtype = dset['objval'].dtype)
        obj2 = dset.create_dataset('SCAmax', shape = dset['objval'].shape, fillvalue = np.nan, dtype = dset['objval'].dtype)
        popt2 = dset.create_dataset('SCAmax_xopt', shape = dset['objval'].shape + (4,), fillvalue = np.nan, dtype = dset['objval'].dtype)
    except RuntimeError:
        obj = dset['SCA']
        popt = dset['SCA_xopt']
        obj2 = dset['SCAmax']
        popt2 = dset['SCAmax_xopt']
    
    
    pt = None
    for cidx, pidx in pb.progressbar(it.product(range(11907,20000), range(obj.shape[1])), widget = pb.ETA, max_value = (20000-11907)*obj.shape[1]):
        h = np.asarray(dset['input/channel_to_noise_matched'][cidx], dtype = float)
        p = Plin[pidx]

        if pidx == 0:
            pt = None

        if pt is not None:
            o1,p1 = SCA(h, mu, Pc, Pmax = p, pt = pt)
        else:
            o1 = -np.inf

        o2,p2 = SCA(h, mu, Pc, Pmax = p, pt = np.full(4, p))
        obj2[cidx,pidx] = o2
        popt2[cidx,pidx,:] = p2

        if o1 > o2:
            obj[cidx,pidx] = o1
            pt = p1
        else:
            obj[cidx,pidx] = o2
            pt = p2

        popt[cidx,pidx,:] = pt


    f.close()
