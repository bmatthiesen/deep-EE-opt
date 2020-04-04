#!/usr/bin/env python3

"""Evaluate model against baseline"""

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
import matplotlib.pyplot as plt
import pandas as pd
import os.path

dfn = '../../data/dset4.h5'
rfn = '../../data/wsee4-processed.h5'
rdb = '../../results/final'
rdn = ['128', '32', '16']
hbn = 'history-wp{wpidx}.h5'
mbn = 'model-wp{wpidx}.h5'
lbn = 'loss-wp{wpidx}.h5'
tbn = 'tloss-wp{wpidx}.h5'

nwp = 10

mu = 4
Pc = 1
B = 180e3

scale = 1e-6 * B * np.log2(np.e) # convert to Mbit

import statsmodels.api as sm
def ecdf(data, logtransform = True):
    if logtransform:
        edges = 10**np.histogram_bin_edges(np.log10(data), bins='auto')
    else:
        edges = np.histogram_bin_edges(data, bins='auto')

    cdf = sm.distributions.ECDF(data)

    return (edges, cdf(edges))

def wsee_pmax(hin):
    h = 10**np.asarray(hin[:,:-1],dtype=float).reshape(hin.shape[0], 4, 4)
    p = 10**np.asarray(hin[:,-1], dtype=float)

    direct = np.diagonal(h, axis1=1, axis2=2)
    ifn = 1+np.sum(h,axis=2)-direct
    rates = np.log(1+direct/ifn)
    ee = rates/(mu*p[:,np.newaxis]+Pc)
    wsee = np.sum(ee, axis=-1)

    return wsee

def wsee_best(hin):
    h = 10**np.asarray(hin[:,:-1],dtype=float).reshape(hin.shape[0], 4, 4)
    p = 10**np.asarray(hin[:,-1], dtype=float)

    direct = np.diagonal(h, axis1=1, axis2=2)
    best = np.max(direct, axis=-1)

    rates = np.log(1+best)
    ee = rates/(mu*p+Pc)

    return ee

def predict(model, inp):
    pred = np.asarray(model.predict(inp), dtype=np.float)
    predlin = K.clip(tf.Variable(10**pred), 0, 1)
    Pmax = 10**np.array(inp[:,-1])
    obj = K.eval(dl.calcObjective([tf.Variable(10**np.asarray(inp, dtype=np.float)), predlin]))

    """
    if obj.shape[0] % PdB.shape[0] != 0:
        N = obj.shape[0] - obj.shape[0] % PdB.shape[0]
        obj = obj[:N]
    else:
        N = obj.shape[0]
    """

    return obj

if __name__=="__main__":
    def reshape(d):
        return np.nanmean(scale * d.reshape(int(d.shape[0]/len(PdB)), len(PdB)), axis=0)

    with h5py.File(rfn, 'r') as f:
        PdB = f['input/PdB'][...]
        wsee = pd.DataFrame(index=PdB)

    for r in rdn:
        rd = rdb + r

        with h5py.File(dfn,'r') as f:
            hin = f['test/input'][...]

            opt = f['test/objval'][...]

            ann = np.full((nwp, len(PdB)), np.nan)
            relerr = np.full((nwp, len(f['test/input'])), np.nan)
            for i in range(nwp):
                ANN = predict(keras.models.load_model(os.path.join(rd, mbn.format(wpidx=i))), f['test/input'][...])
                ann[i,:] = reshape(opt)
                relerr[i,:] = np.abs(opt - ANN) / opt # scaling is not necessary here

            relerr = np.mean(relerr, axis=0)

            wsee = wsee.assign(
                    opt = reshape(opt),
                    SCA = reshape(f['test/SCA'][...]),
                    SCAos = reshape(f['test/SCAmax'][...]),
                    ANN = np.mean(ann, axis=0)
                    )

            x, y = ecdf(relerr)
            relerr2 = pd.DataFrame(index = x, data = {'y': y})
            del y

        wsee = wsee.assign(
                max = reshape(wsee_pmax(hin)),
                best = reshape(wsee_best(hin))
                )

        # collect losses
        loss = None
        val_loss = None
        for i in range(nwp):
            with h5py.File(os.path.join(rd, tbn.format(wpidx=i)), 'r') as f:
                if loss is None:
                    loss = np.full((nwp, f['training_loss'].shape[0]), np.nan)
                    val_loss = loss.copy()

                loss[i] = f['training_loss'][:]

            with h5py.File(os.path.join(rd, lbn.format(wpidx=i)), 'r') as f:
                val_loss[i] = f['loss'][:]

        loss = pd.DataFrame(data = {'loss': np.mean(loss, axis=0), 'val_loss': np.mean(val_loss, axis=0)})
        loss.index += 1
        del val_loss


        # save
        wsee.to_csv('../tex/verification_%s.dat' % r, index_label = 'snr')
        relerr2.to_csv('../tex/verification_relerr_%s.dat' % r, index_label = 'x')
        loss.to_csv('../tex/training_%s.dat' % r, index_label = 'epoch')

        #wsee.plot()
        #plt.show()
