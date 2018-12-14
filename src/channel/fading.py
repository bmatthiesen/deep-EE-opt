#!/usr/bin/env python3

# Copyright (C) 2018 Bho Matthiesen
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah,
# "Deep Learning for Optimal Energy-Efficient Power Control in Wireless
# Interference Networks," submitted to IEEE Journal on Selected Areas in
# Communication.
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

def shadowing(stddev, shape = None):
    """Log-Normal Shadow Fading [1, Sect. 3.9.2]
    Parameter:
        stddev: Standard Deviation \sigma in dB
        shape: shape of generated random matrix

    Suitable Parameter [1, Sect. 3.10.7]:
        Outdoor Scenario: 7 - 9 dB

    Returns:
        Matrix with Fading in dB

    References:
        [1] T. S. Rappaport, "Wireless Communications: Principles and Practices."
            Prentice Hall, 2nd ed.
    """

    return np.random.normal(0.0, stddev, shape)

C1 = .5*np.sqrt(3)
F1 = np.array([[C1, 0],[-.5, 1]])
F2 = np.array([[-C1, C1], [-.5, -.5]])
F3 = np.array([[0, -C1], [1, -.5]])

def uniformHexagon(num):
    """Genrates num points in unit Hexagon

    Uses Algorithm from [1, Appendix G]

    References:
        [1] T. L. Marzetta, E. G. Larsson, H. Yang, and H. Q. Ngo,
            "Fundamentals of Massive MIMO," Cambridge University Press, 2016.
    """

    tmp = np.random.rand(num)

    K1_sel = tmp < 1/3
    K2_sel = np.logical_and(tmp>=1/3, tmp<2/3)
    K3_sel = tmp >= 2/3

    K1 = np.sum(K1_sel)
    K2 = np.sum(K2_sel)
    K3 = np.sum(K3_sel)

    ret = np.empty((num, 2))

    if K1 > 0:
        tmp = np.random.rand(K1, 2)
        ret[K1_sel] = np.array([F1 @ x for x in tmp])

    if K2 > 0:
        tmp = np.random.rand(K2, 2)
        ret[K2_sel] = np.array([F2 @ x for x in tmp])

    if K3 > 0:
        tmp = np.random.rand(K3, 2)
        ret[K3_sel] = np.array([F3 @ x for x in tmp])

    return ret

#import matplotlib.pyplot as plt
#def plot(p):
#    fig, ax = plt.subplots()
#    ax.scatter(p[:,0],p[:,1])
#    plt.show()

def crandn(*args, **kwargs):
    return 1/np.sqrt(2) * (np.random.randn(*args, **kwargs) + 1j * np.random.randn(*args, **kwargs))

def crandn_corr(var):
    var = .5 * np.asarray(var)
    mean = [0.]*var.shape[0]

    return np.random.multivariate_normal(mean, var) + 1j * np.random.multivariate_normal(mean, var, check_valid='raise')

def rayleigh(shape, shadowing_stddev, PL):
    """Rayleigh Fading Channel Coefficients

    Parameters:
        shape: Tuple with shape of output matrix
        shadowing_stddev: Standard Deviation \sigma in dB for log-normal
                          shadowing (None for no shadowing)
        PL: Path Loss in dB

    Returns:
        Channel coefficient h
        Large Scale Pathloss in dB
    """
    beta_dB = PL

    if shadowing_stddev is not None:
        beta_dB = shadowing(shadowing_stddev) + beta_dB

    beta = 10**(-beta_dB/10)
    h = crandn(*shape)

    return (np.sqrt(beta) * h, beta_dB)

def corr_rayleigh(PL, dist, Nt, corr):
    beta_dB = PL(dist)
    beta = 10**(-beta_dB/10)

    shape = dist.shape + (Nt,)
    num = np.prod(shape)
    Q = np.full((num,num), corr)
    Q[np.diag_indices(num)] = 1

    h = crandn_corr(Q)
    h = h.reshape(shape)

    return (np.expand_dims(np.sqrt(beta), 2) * h, beta_dB)
