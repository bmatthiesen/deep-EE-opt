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

import os.path
import h5py
from pathlib import Path
import sys
import numpy as np

respath = '/home/matthiesen/Work/deep-opt/tmp/'
outfile = os.path.join(respath, 'results.h5')
chanfile = '/home/matthiesen/Work/deep-opt/data/channels-7.h5'
#respath = '../../results/'
#outfile = os.path.join(respath, 'results.h5')
#chanfile = '../../data/channels.h5'

glob_pattern = "res_channel*h5"

scale = np.log2(np.e)

with h5py.File(chanfile, 'r') as f:
    numChan = f['input']['channel'].shape[0]

p = Path(respath)

with h5py.File(outfile, 'w') as outf:
    first = True
    for fn in p.glob(glob_pattern):
        try:
            with h5py.File(str(fn),'r') as res:
                if first:
                    print('Initializing')

                    first = False

                    objval = outf.create_dataset('objval', (numChan, res['results'].shape[0]), dtype = res['results']['Objective Value'].dtype, fillvalue = np.nan)
                    xopt = outf.create_dataset('xopt', (numChan, res['results'].shape[0]), dtype = (res['results']['xopt'].dtype, res['results']['xopt'].shape[1]))
                    runtime = outf.create_dataset('runtime', (numChan, res['results'].shape[0]), dtype = res['results']['Runtime'].dtype, fillvalue = np.nan)

                    dset = outf.create_group('input')
                    dset.create_dataset('PdB', data = res['PdB'], dtype = res['PdB'].dtype)

                    with h5py.File(chanfile, 'r') as cf:
                        dset.create_dataset('Channels', data = cf['input']['channel'])
                        dset.create_dataset('channel_to_noise_matched', data = cf['input']['channel_to_noise_matched'])

                    dset.create_dataset('PA inefficency',data = 4.0, dtype = np.float32)
                    dset.create_dataset('Pc',data = 1.0, dtype = np.float32)
                    dset.create_dataset('epsilon', data = res['results'][0]['Epsilon'], dtype=res['results']['Epsilon'].dtype)
                    dset.create_dataset('Relative Tolerance', data = res['results'][0]['Relative Tolerance'], dtype = np.bool)

                robjval = scale * res['results']['Objective Value'][...]
                rxopt = res['results']['xopt'][...]
                rruntime = res['results']['Runtime'][...]

                sel = res['results']['Status'] != 'Optimal'
                if np.any(sel):
                    robjval[sel] = np.nan
                    rxopt[sel] = np.nan
                    rruntime[sel] = np.nan
                
                objval[res['cidx'][...], :] = robjval
                xopt[res['cidx'][...], :] = rxopt
                runtime[res['cidx'][...], :] = rruntime

                print(str(fn))

        except OSError as e:
            print('{}: OSError: {}'.format(fn,e), file=sys.stderr)

    if first:
        print('No result files found!')
        sys.exit(1)

    print('Finished reading. Now calculating some more values ...')

    h = np.asarray(outf['input']['channel_to_noise_matched'][...], dtype=float)
    Plin = 10**(np.asarray(outf['input']['PdB'][...], dtype=float)/10)
    xo = np.asarray(xopt[...], dtype=float)
    mu = outf['input']['PA inefficency']
    Pc = outf['input']['Pc']

    sr = np.full((len(Plin), h.shape[0]), np.nan)
    sr2 = sr.copy()
    ee2 = sr.copy()

    for i in range(len(Plin)):
        # max power
        p = h*Plin[i]
        direct = np.diagonal(p, axis1=1, axis2=2)
        ifn = 1+np.sum(p,axis=2)-direct

        rates = np.log2(1+direct/ifn)
        sr[i] = np.sum(rates,axis=1)

        # wsee optimal sr
        p = (h.swapaxes(0,1) * xo[:,i]).swapaxes(1,0)
        direct = np.diagonal(p, axis1=1, axis2=2)
        ifn = 1+np.sum(p,axis=2)-direct

        rates = np.log2(1+direct/ifn)
        sr2[i] = np.sum(rates,axis=1)
        ee2[i] = np.sum(rates / (mu*xo[:,i]+Pc), axis=1)

    sr = sr.T
    ee = sr / (mu*Plin+Pc)

    dset = outf.create_group('max power')
    dset.create_dataset('sumrate', data = sr, dtype=np.float32)
    dset.create_dataset('wsee', data = ee, dtype=np.float32)
    outf.create_dataset('sumrate', data = sr2.T, dtype=np.float32)
    outf.create_dataset('wsee', data = ee2.T, dtype=np.float32)
