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

import h5py
import numpy as np

ofn = '../../data/dset_final.h5'
f = h5py.File('../../data/wsee4-processed.h5', 'r')  # 4 user scenario
#f = h5py.File('../../data/wsee7-processed.h5', 'r')  # 7 user scenario

sets = [
        {'name': 'validation', 'cidx': slice(200), 'pidx': slice(None)},
        {'name': 'training', 'cidx': slice(1000, 3000), 'pidx': slice(None)},
        {'name': 'test', 'cidx': slice(10000, 20000), 'pidx': slice(None)},
        ]

Plin = 10**(np.asarray(f['input']['PdB'][...], dtype=np.float)/10)

with h5py.File(ofn, 'w') as outf:
    for s in sets:
        c = np.multiply.outer(np.asarray(f['input/channel_to_noise_matched'][s['cidx'], s['pidx']], dtype=np.float),Plin, dtype=np.float)
        c = np.moveaxis(c, (1, 2, 3), (2, 3, 1))
        c = c.reshape(*c.shape[:2], np.prod(c.shape[2:]))

        pp = np.expand_dims(np.repeat(Plin[np.newaxis], c.shape[0], axis = 0), axis=-1)
        c = np.concatenate((c, pp), axis=2)
        c = c.reshape(np.prod(c.shape[:2]), c.shape[2])
        c = np.log10(c)
        assert(not np.any(np.isnan(c)))

        outf.create_group(s['name'])
        outf[s['name']].create_dataset('input', data = c, dtype = np.float32)

        del c, pp


        x = (np.asarray(f['xopt'][s['cidx'], s['pidx']], dtype=np.float).swapaxes(1,2) / Plin).swapaxes(1,2)
        x = x.reshape(np.prod(x.shape[:2]), np.prod(x.shape[2:]))
        with np.errstate(divide='ignore'):
            x = np.log10(x)
        x[~np.isfinite(x)] = -20

        outf[s['name']].create_dataset('xopt', data = x, dtype = np.float32)

        del x


        # convert objval to nats and reshape for keras
        o = np.asarray(f['objval'][s['cidx'], s['pidx']],dtype=np.float)
        o = np.log(2) * np.reshape(o, np.prod(o.shape))

        outf[s['name']].create_dataset('objval', data = o, dtype = np.float32)

        del o

        try:
            sca = np.asarray(f['SCA'][s['cidx'], s['pidx']])
            scam = np.asarray(f['SCAmax'][s['cidx'], s['pidx']])
        except KeyError:
            sca = None
            scam = None

            print('Skipping SCA')

        if sca is not None:
            Bsca = np.isnan(sca)

            if np.all(~Bsca):
                outf[s['name']].create_dataset('SCA', data = np.reshape(sca, np.prod(sca.shape)))
                outf[s['name']].create_dataset('SCAmax', data = np.reshape(scam, np.prod(scam.shape)))
            elif np.any(~Bsca):
                print('SCA looks wrong')

