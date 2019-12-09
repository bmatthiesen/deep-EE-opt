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

Config = (
        #('sum_ann512_512_elu_relu', 512, True, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'elu', 'relu', 'elu', 'relu'])),
        #('ann512_512_elu_relu', 512, False, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'elu', 'relu', 'elu', 'relu'])),
        #
        #('ann512_512_relu', 512, False, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #('sum_ann512_512_relu', 512, True, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #
        #('ann512_512_elu_start', False, 512, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #('ann512_512_elu_end', False, 512, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        #('ann512_512_elu_start_and_end', False, 512, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        ##
        #('sum_ann512_512_elu_start', True, 512, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #('sum_ann512_512_elu_end', True, 512, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        #('sum_ann512_512_elu_start_and_end', True, 512, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        ##
        #('ann512_32_elu_relu', 32, False, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'elu', 'relu', 'elu', 'relu'])),
        #('sum_ann512_32_elu_relu', 32, True, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'elu', 'relu', 'elu', 'relu'])),
        #('ann512_32_relu', 32, False, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #('sum_ann512_32_relu', 32, True, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        ##
        #('ann512_32_elu_start', False, 32, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #('ann512_32_elu_end', False, 32, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        #('ann512_32_elu_start_and_end', False, 32, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        ##
        #('sum_ann512_32_elu_start', True, 32, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'relu'])),
        #('sum_ann512_32_elu_end', True, 32, ([512, 256, 128, 64, 32, 16], ['relu', 'relu', 'relu', 'relu', 'relu', 'elu'])),
        #('sum_ann512_32_elu_start_and_end', True, 32, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'relu', 'relu', 'relu', 'elu'])),

        #Karl
        #('k_ann1024_32_256_relu_adam', 256, False, ([1024, 512, 256, 128, 64, 32, 32], ['relu']*7)),
        #('k_ann1024_1024_256_relu_elu_adam', 256, False, ([1024, 1024, 512, 256, 128, 64, 32, 16], ['elu']+['relu']*7)),
        #('k_ann1024_2048_relu_adam', 256, False, ([1024, 2048, 512, 256, 128, 64, 32, 16], ['relu']*8)),
        #('k_ann1024_1024_256_relu_adam', 256, False, ([1024, 1024, 512, 256, 128, 64, 32, 16], ['relu']*8)),
        #('k_ann1024_256_relu_perm_layer_adam_new', 256, False, ([1024, 512, 256, 128, 64, 32, 16], ['relu']*7)),
        #('k_ann4096_256_elu_relu_elu_perm_layer_adam_5k', 256, False, ([4096, 2048, 1024, 512, 256, 128, 64, 32], ['elu'] + ['relu']*6 + ['elu'])),
        #('k_ann2048_16_256_elu_relu_elu_perm_layer_adam_5k', 256, False, ([2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*6 + ['elu'])),
        #('k_ann2048_16_512_elu_relu_elu_perm_layer_adam_5k', 512, False, ([2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*6 + ['elu'])),
        #('k_ann1024_2048_16_256_elu_relu_elu_perm_layer_adam_5k', 256, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_2048_16_128_relu_perm_layer_adam_7k', 128, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16], ['relu']*9)),
        #('k_ann1024_2048_16_128_elu_relu_elu_perm_layer_adam_7k', 128, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_2048_16_256_elu_relu_perm_layer_adam_7k', 256, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*8)),
        #('k_ann1024_2048_16_512_elu_relu_perm_layer_adam_7k', 512, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*8)),
        #('k_ann1024_2048_16_512_elu_relu_elu_perm_layer_adam_7k', 512, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_2048_7_512_elu_relu_elu_perm_layer_adam_7k', 512, False, ([1024, 2048, 1024, 512, 256, 128, 64, 32, 16, 7], ['elu'] + ['relu']*8 + ['elu'])),
        #('k_ann1024_4096_16_512_elu_relu_elu_perm_layer_adam_7k', 512, False, ([1024, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_4096_16_768_elu_relu_elu_perm_layer_adam_7k', 768, False, ([1024, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_4096_16_1024_elu_relu_elu_perm_layer_adam_7k5', 1024, False, ([1024, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_5120_16_768_elu_relu_elu_perm_layer_adam_7k5', 768, False, ([1024, 5120, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_4096_16_768_elu_relu_elu_perm_layer_adam_7k5', 768, False, ([1024, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])), #the best
        #('k_ann1024_5120_16_1024_elu_relu_elu_perm_layer_adam_7k5', 1024, False, ([1024, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann1024_4096_16_512_elu_relu_perm_layer_adam_7k', 512, False, ([1024, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*8)),
        #('k_ann2048_4096_16_256_elu_relu_elu_perm_layer_adam_5k', 256, False, ([2048, 4096, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('k_ann4096_16_256_elu_relu_elu_perm_layer_adam_5k', 256, False, ([4096, 2048, 1024, 512, 256, 128, 64, 32, 16], ['elu'] + ['relu']*7 + ['elu'])),
        #('ann512_32_elu_relu', 32, False, ([512, 256, 128, 64, 32, 16], ['elu', 'relu', 'elu', 'relu', 'elu', 'relu'])),
        #('4users_1perc', 0.01, 32, False, ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])),
        #('4users_5perc', 0.05, 32, False, ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])),
        #('4users_rerun', 1., 32, False, ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])),
        ('4users_rerun', 1., 128, False, ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])),
    )

numReal = 10
nEpochs = 500

def doPlot(dfile, results, savedir, trainOnObj, sumLayer, epoch = None):
    # set result paths

    if epoch is None:
        mfilename = 'model-wp{}.h5'
        plotfilename = 'plot-wp{}.png'
    else:
        mfilename = 'modelCP-wp{}.%03d.h5' % (epoch, )
        plotfilename = 'plot-wp{}.%03d.png' % (epoch, )

    histfile = os.path.join(savedir, 'history-wp{}.h5'.format(wpidx))
    mfile = os.path.join(savedir, mfilename.format(wpidx))

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

    model1 = dl.createModel(layer, trainOnObj, sumLayer)
    model1.load_weights(mfile)

    """
    model2 = dl.createModel(layer, trainOnObj)
    model2.load_weights(mfile2)
    """

    # get PdB
    with h5py.File(results, 'r') as rf:
        PdB = rf['input/PdB'][...]

        sca = np.array(rf['SCA'])
        sca_max = np.array(rf['SCAmax'])

    def predict(model, dset):
        pred = np.asarray(model.predict(dset['input']), dtype=np.float)

        if sumLayer:
            pred = pred[:,:-1]

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

            #a.plot(PdB, np.mean(sca, axis=0).T, label='SCA')
            #a.plot(PdB, np.mean(sca_max, axis=0).T, label='SCA max')

            a.set(title = s, xlabel='PdB', ylabel='WSEE')
            a.grid(True)
            a.legend()

    # show plot
    fig.suptitle(' -- '.join(['{}: {}'.format(s, a) for s,a in zip(*layer)]))
    fig.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(savedir, plotfilename.format(wpidx)))

if __name__=="__main__":
    for savedir, percent, batchsize, sumLayer, layer in Config:
        for wpidx in range(numReal):
            # configure
            trainOnObj = False
            dfile = 'dset4.h5'
            results = 'wsee4-processed.h5'
            #dfile = 'dset7.h5'  # 7 users
            #results = 'wsee7-processed.h5'  # 7 users

            #if batchsize == 512:
            #    #nEpochs = 7500
            #    nEpochs = 7500
            #else:
            #    #nEpochs = 300
            #    nEpochs = 500
            #    #nEpochs = 7500

            # train
            try:
                dl.DL(layer, batchsize, nEpochs, dfile, savedir, wpidx, trainOnObj, sumLayer, perc_tset=percent)

                doPlot(dfile, results, savedir, trainOnObj, sumLayer)

                #for e in range(100, nEpochs, 100):
                #    doPlot(dfile, results, savedir, trainOnObj, sumLayer, epoch = e)
            except Exception as e:
                raise
                print(e)

