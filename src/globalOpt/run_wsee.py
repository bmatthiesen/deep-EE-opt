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
taskid = int(os.getenv("SLURM_ARRAY_TASK_ID"))
savedir = os.getenv('JOB_HPC_SAVEDIR')

# create necessary directories
os.makedirs(savedir, exist_ok = True)

# set filenames
basename = 'task{}.h5'.format(taskid)

try:
    wpfile = sys.argv[1]
except IndexError:
    wpfile = '../../data/channels.h5'

outfile = os.path.join(savedir, 'res_' + basename)

# open h5 files
chanf = h5py.File(wpfile, 'r')
heff = chanf['input']['channel_to_noise_matched']


# compute channel list
numtasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
channel_ids = np.array_split(np.arange(heff.shape[0], dtype=np.uint), numtasks)[taskid]

# create result file
result_dt = wseePy.WSEE(1,1).getResultDt()

with h5py.File(outfile, 'w') as resf:
    resf.create_dataset('channel indices', data = channel_ids)
    resf.create_dataset('PdB', data = PdB)
    resf.create_dataset('results', (len(channel_ids), len(PdB)), dtype=result_dt, chunks = (1, len(PdB)))

# some debug output
print('Task ID = {}\nSavefile = {}\nNum Tasks = {}\nChannel IDs = {}\n\n'.format(taskid, outfile, numtasks, str(channel_ids)))

# start slaving away
dix = np.diag_indices(4)
t = wseePy.WSEE(mu, Pc)
for idx in range(len(channel_ids)):
    cidx = channel_ids[idx]

    print(30*'=' + ' cidx = ' + str(cidx) + ' ' + 30*'=')
    sys.stdout.flush()

    res = np.empty(len(PdB), dtype=result_dt)

    beta = np.asarray(heff[cidx], dtype=np.double)
    alpha = beta[dix]
    beta[dix] = 0
    t.setChan(alpha, beta)

    for pidx in range(len(PdB)):
        t.setPmax(Plin[pidx])
        t.optimize()

        res[pidx] = t.result()

    with h5py.File(outfile, 'r+') as resf:
        resf['results'][idx,:] = res

print(30*'=' + ' DONE ' + 30*'=')

resf.close()
chanf.close()
