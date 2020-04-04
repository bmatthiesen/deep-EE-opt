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

import h5py
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import dl
from evaluate import rdn, rdb, mbn, nwp, ecdf
import pandas as pd
import matplotlib.pyplot as plt
import os.path

rfn = {
        'hataUrban_noSF': '../../data/results_hataUrban_noSF.h5',
        'hataUrban': '../../data/results_hataUrban.h5',
    }

sl = slice(None)

B = 180e3
scale = 1e-6 * B

for r in rdn:
    rd = rdb + r

    def predict(inp):
        obj = None

        for i in range(nwp):
            model = keras.models.load_model(os.path.join(rd, mbn.format(wpidx=i)))
            pred = np.asarray(model.predict(inp), dtype=np.float)

            if obj is None:
                obj = np.full((nwp, len(pred)), np.nan)

            predlin = K.clip(tf.Variable(10**pred), 0, 1)
            obj[i,:] = K.eval(dl.calcObjective([tf.Variable(10**np.asarray(inp, dtype=np.float)), predlin]))

        return obj

    wsee = None
    relerr = dict()
    for n,fn in rfn.items():
        with h5py.File(fn, 'r') as f:
            PdB = f['input']['PdB'][...]
            Plin = 10**(np.asarray(PdB, dtype=np.float)/10)

            if wsee is None:
                wsee = pd.DataFrame(index=PdB)
            else:
                assert(np.all(wsee.index == PdB))

            c = f['input/channel_to_noise_matched'][sl]
            c = np.multiply.outer(c, Plin)
            c = np.moveaxis(c, (1, 2, 3), (2, 3, 1))
            c = c.reshape(*c.shape[:2], np.prod(c.shape[2:]))

            pp = np.expand_dims(np.repeat(Plin[np.newaxis], c.shape[0], axis = 0), axis=-1)
            c = np.concatenate((c, pp), axis=2)
            c = c.reshape(np.prod(c.shape[:2]), c.shape[2])
            c = np.log10(c)

            del pp

            o = predict(c)

            # convert to bits
            o *= np.log2(np.e)

            # convert objval
            oo = np.asarray(f['objval'][sl],dtype=np.float)

            if (np.any(np.isnan(oo))):
                print('Warning: NaN in ' + n)

            o_mean = scale * np.mean(np.mean(o.reshape(o.shape[0], int(o.shape[1]/len(PdB)), len(PdB)), axis=1), axis = 0)
            oo_mean = scale * np.nanmean(oo, axis=0)

            # put in data frame
            wsee = wsee.assign(**{n + '_opt': oo_mean, n + '_ANN': o_mean})

            # relerr CDF
            re = np.full(o.shape, np.nan)
            for i in range(o.shape[0]):
                re[i,:] = np.abs(oo.flatten() - o[i]) / oo.flatten() # scaling is not necessary here
            re = np.mean(re, axis=0)

            x, y = ecdf(re)

            relerr[n] = pd.DataFrame(index = y, data = {'x': x})

    fig, (ax1, ax2) = plt.subplots(1,2)
    wsee.plot(ax = ax1)

    wsee.to_csv('../tex/generalization_%s.dat' % r, index_label = 'snr')

    for n in relerr:
        relerr[n].to_csv('../tex/generalization_relerr_{}_{}.dat'.format(n,r), index_label = 'y')
        ax2.semilogx(relerr[n], relerr[n].index, label = n)

    """
    ax2.legend()
    plt.show()
    """
