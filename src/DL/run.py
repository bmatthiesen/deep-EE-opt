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

import dl
import os.path
import pathlib
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import keras
import tensorflow as tf
import keras.backend as K

SDs = (
        '../../results/final128',
        '../../results/final16',
    )
Ls = (
        ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu']),
        ([16, 8], ['elu', 'relu']),
    )

numReal = 10
nEpochs = 100

if __name__=="__main__":
    for savedir, layer in zip(SDs, Ls):
        for wpidx in range(numReal):
            # configure
            trainOnObj = False
            dfile = '../../data/dset_final.h5'
            results = '../../data/results.h5'

            # train
            dl.DL(layer, 128, nEpochs, dfile, savedir, wpidx, trainOnObj)

            # set result paths
            histfile = os.path.join(savedir, 'history-wp{}.h5'.format(wpidx))
            mfile = os.path.join(savedir, 'model-wp{}.h5'.format(wpidx))

            """
            mfile2 = None
            mepoch = -1
            mloss = np.inf
            for fn in pathlib.Path(savedir).glob('modelCP-wp0.*'):
                match = re.search('-wp0.(\d+)-([0-9.]+)\.h5', fn.name)
                epoch = match.group(1)
                loss = match.group(2)

                if mfile2 is None or (epoch > mepoch and loss <= mloss):
                    mepoch = epoch
                    mloss = loss
                    mfile2 = fn.name
            mfile2 = os.path.join(savedir, mfile2)
            """

            # prepare plot
            fig, ax = plt.subplots(2,2, figsize = (13,8), dpi = 96)

            # plot history
            with h5py.File(histfile,'r') as f:
                ax[0,0].plot(f['loss'], label='loss')
                ax[0,0].plot(f['val_loss'], label='val_loss')
                ax[0,0].set(title='training process', xlabel = 'epoch', ylabel = 'mse')
                ax[0,0].grid(True)
                ax[0,0].legend()

            # load models


            #model1 = keras.models.load_model(mfile, custom_objects={"tf": tf})
            #model2 = keras.models.load_model(mfile2, custom_objects={"tf": tf}

            model1 = dl.createModel(layer, trainOnObj)
            model1.load_weights(mfile)

            """
            model2 = dl.createModel(layer, trainOnObj)
            model2.load_weights(mfile2)
            """

            # get PdB
            with h5py.File(results, 'r') as rf:
                PdB = rf['input/PdB'][...]

            def predict(model, dset):
                pred = np.asarray(model.predict(dset['input']), dtype=np.float)

                if trainOnObj:
                    obj = pred
                else:
                    predlin = K.clip(tf.Variable(10**pred), 0, 1)
                    Pmax = 10**np.array(dset['input'][:,-1])
                    obj = K.eval(dl.calcObjective([tf.Variable(10**np.asarray(dset['input'][...], dtype=np.float)), predlin]))

                if obj.shape[0] % PdB.shape[0] != 0:
                    N = obj.shape[0] - obj.shape[0] % PdB.shape[0]
                    obj = obj[:N]
                else:
                    N = obj.shape[0]

                obj = obj.reshape(int(obj.shape[0]/PdB.shape[0]), PdB.shape[0])

                return (pred, obj, N)

            # plot WSEEs
            with h5py.File(dfile, 'r') as d:
                for s, a in zip(('training', 'validation', 'test'), (ax[0,1], *ax[1])):
                    print('predicting on {} set'.format(s))

                    dset = d[s]
                    _, obj, N = predict(model1, dset)
                    #_, obj2, _ = predict(model2, dset)

                    objopt = np.asarray(dset['objval'][:N]).reshape(*obj.shape)

                    a.plot(PdB, np.mean(objopt, axis=0).T, label='optimal')
                    a.plot(PdB, np.mean(obj,axis=0).T, label='final model')
                    #a.plot(PdB, np.mean(obj2,axis=0).T, ':', label='min(val_loss)')
                    a.set(title = s, xlabel='PdB', ylabel='WSEE')
                    a.grid(True)
                    a.legend()

            # show plot
            fig.suptitle(' -- '.join(['{}: {}'.format(s, a) for s,a in zip(*layer)]))
            fig.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(savedir, 'plot-wp{}.png'.format(wpidx)))
