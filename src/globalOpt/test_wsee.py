#!/usr/bin/env python3

# Copyright (C) 2018-2019 Bho Matthiesen, Karl-Ludwig Besser
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
import wseePy

beta = np.array([[8.3401758e+02, 5.9968562e+00, 9.5184622e+00, 6.0737956e-01],
[1.3587096e+00, 3.9182301e+01, 2.0014184e-02, 1.6249435e+00],
[3.8521406e-01, 4.6761915e-01, 8.7457578e+03, 1.8704400e+00],
[1.2729254e-01, 2.1447293e-02, 3.1017335e-02, 1.2471862e+02]])



dix = np.diag_indices(4)
alpha = beta[dix]
beta[dix] = 0

w = wseePy.WSEE4(4, 1)
w.setPmax(1e2)
w.setChan(alpha, beta)
w.optimize()
