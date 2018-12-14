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

import subprocess
import sys
import os.path
import h5py
import numpy as np
from run import numReal, SDs, nEpochs
import itertools as it

bn = 'modelCP-wp{wpidx}.{epoch:03d}.h5'
indices = range(1,nEpochs+1)

gn = 'training_loss'
for bp, wp in it.product(SDs, range(numReal)):
    ofn = os.path.join(bp, 'tloss-wp{}.h5'.format(wp))
    with h5py.File(ofn, 'w') as f:
        f.create_dataset(gn, dtype = np.float, shape = (len(indices), ), fillvalue = np.nan)

    for idx in indices:
        print('Epoch = %d'%idx)
        subprocess.call([sys.executable, 'val_loss.py', os.path.join(bp, bn).format(epoch = idx, wpidx = wp), ofn, gn, str(idx-1)])
