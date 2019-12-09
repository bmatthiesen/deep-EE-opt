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

import wseePy
import numpy as np
import h5py
import os
import os.path
import sys
import atexit

# parameters
mu = 4.0
Pc = 1.0
PdB = np.array(range(-30,20+1,1))

Plin = 10**(PdB/10)

# load config
cidx = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
savedir = os.getenv('JOB_HPC_SAVEDIR', "tmp")

# create necessary directories
os.makedirs(savedir, exist_ok = True)

# set filenames
basename = 'channel{}.h5'.format(cidx)

try:
    wpfile = sys.argv[1]
except IndexError:
    wpfile = '../../data/channels-7.h5'

outfile = os.path.join(savedir, 'res_' + basename)

# open h5 files
chanf = h5py.File(wpfile, 'r')
heff = chanf['input']['channel_to_noise_matched']

# get dimension and set select correct WSEE solver
numUE = heff.shape[1]
WSEE = getattr(wseePy, 'WSEE%d' % numUE)

# create result file
result_dt = WSEE(1,1).getResultDt()

try:
    with h5py.File(outfile, 'x') as resf:
        resf.create_dataset('cidx', data = cidx)
        resf.create_dataset('Channel', data = chanf['input']['channel'][cidx])
        resf.create_dataset('Effective Channel', data = heff[cidx])
        resf.create_dataset('PdB', data = PdB)
        resf.create_dataset('results', (len(PdB),), dtype=result_dt)

    pidxRange = range(len(PdB))
except OSError as e:
    with h5py.File(outfile, 'r') as resf:
        if resf['cidx'][...] != cidx:
            raise RuntimeError('corrupted result file (cidx mismatch)')
        if resf['results'].shape[0] != len(PdB):
            raise RuntimeError('corrupted result file (PdB len mismatch)')
        
        pidxRange = np.where(resf['results']['Epsilon'] == 0)[0]


# some debug output
print('Channel ID = {}\nSavefile = {}\nNum Tasks = {}'.format(cidx, outfile, len(PdB)))

if len(pidxRange) != len(PdB):
    print('Remaining Tasks = {}\n\n'.format(len(pidxRange)))
else:
    print('\n\n')
sys.stdout.flush()

# start slaving away
dix = np.diag_indices(numUE)
t = WSEE(mu, Pc)

beta = np.asarray(heff[cidx], dtype=np.double)
alpha = beta[dix]
beta[dix] = 0
t.setChan(alpha, beta)

for pidx in pidxRange:
    print((30*'=' + ' PdB = {} ({}/{}) ' + 30*'=').format(PdB[pidx], pidx, len(PdB)))
    sys.stdout.flush()

    t.setPmax(Plin[pidx])
    t.optimize()

    with h5py.File(outfile, 'r+') as resf:
        resf['results'][pidx] = t.result()

    print('\n\n')

print(30*'=' + ' DONE ' + 30*'=')

resf.close()
chanf.close()
