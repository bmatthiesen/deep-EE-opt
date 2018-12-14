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

import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import itertools as it
import h5py
import os
import os.path
import resource
import timeit

def rel_mse(x_true, x_pred):
    loss = K.square(K.abs((x_true - x_pred)/ x_true))
    return K.mean(loss, axis=-1)

mu = 4
Pc = 1

def calcObjective(tensors):
    h = keras.layers.Reshape((4,4))(tensors[0][:,:-1])
    mmu = mu * tensors[0][:,-1]
    x = keras.layers.Activation('relu')(tensors[1])

    o = keras.layers.multiply([h,x])
    alpha = tf.matrix_diag_part(o)
    beta = 1 + (K.sum(o, axis=-1) - alpha)

    rate = K.log(1 + alpha/beta)
    ret = K.sum(rate / (keras.layers.multiply([mmu, x]) + Pc), axis=-1)
    #ret = tf.Print(ret, [ret])
    #ret = K.sum(rate, axis=-1)

    return ret

def calcObjectivePow(tensors):
    ten = tf.constant(10, dtype=tensors[0].dtype)

    h = tf.pow(ten, tensors[0])
    x = tf.pow(ten, tensors[1])

    return calcObjective([h, x])

def createModel(layer, trainOnObj):
    from dl import calcObjective, calcObjectivePow, rel_mse

    inlayer = keras.layers.Input(shape = (17,))

    x = inlayer
    for n, a in zip(*layer):
        x = keras.layers.Dense(n, activation = a)(x)
    predPower = keras.layers.Dense(4, activation = 'linear')(x)

    if trainOnObj:
        objlayer = keras.layers.Lambda(calcObjectivePow, (1,))([inlayer, predPower])
        model = keras.models.Model(inputs = inlayer, outputs = objlayer)
    else:
        model = keras.models.Model(inputs = inlayer, outputs = predPower)

    opt = keras.optimizers.Nadam()

    if trainOnObj:
        model.compile(opt, loss = rel_mse)
    else:
        model.compile(opt, loss ='mean_squared_error')

    return model
def DL(layer, bs, nEpochs, dfile, savedir, wpidx, trainOnObj = True, init = None):
    # create result directory
    os.makedirs(savedir, exist_ok = True)

    # set filenames
    cp_name = os.path.join(savedir, 'modelCP-wp%d.{epoch:03d}.h5' % wpidx)
    history_name = os.path.join(savedir, 'history-wp%d.h5' % wpidx)
    savefile = os.path.join(savedir, 'model-wp%d.h5' % wpidx)


    # build model from WP
    model = createModel(layer, trainOnObj)

    # and show what we did
    model.summary()

    # initialize model with random weights
    if init is None:
        #keras.initializers.RandomNormal() # TODO AZ uses stddev=0.5, Keras doc stddev=0.05
        pass
    else:
        model.load_weights(init)


    keys = ['loss', 'acc', 'val_loss', 'val_acc']
    class LossHistory(keras.callbacks.Callback):
        def __init__(self, fn, overwrite = False):
            self.fn = fn

            if overwrite:
                self.mode = 'w'
            else:
                self.mode = 'w-'

        def on_train_begin(self, logs={}):
            with h5py.File(self.fn, self.mode) as f:
                print('Saving history to {}'.format(self.fn))

                f.create_dataset('runtime', (self.params['epochs'],))

                for k in keys:
                    f.create_dataset(k, (self.params['epochs'],), fillvalue = np.nan)

        def on_epoch_begin(self, epoch, logs={}):
            self.tic = timeit.default_timer()

        def on_epoch_end(self, epoch, logs={}):
            rt = timeit.default_timer() - self.tic

            with h5py.File(self.fn, 'a') as f:
                f['runtime'][epoch] = rt

                for k in keys:
                    try:
                        f[k][epoch] = logs[k]
                    except KeyError:
                        pass

    hist = LossHistory(history_name, True)
    checkpointer = keras.callbacks.ModelCheckpoint(cp_name, verbose=1, save_best_only = False)

    with h5py.File(dfile, 'r') as f:
        tin = f['training/input'][...]

        if trainOnObj:
            tout = f['training/objval'][...]
        else:
            tout = f['training/xopt'][...]

    history = model.fit(tin, tout, validation_data=None, epochs=nEpochs, batch_size=bs, callbacks=[hist, checkpointer])
    model.save(savefile)

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    print(rusage)
    print('Max RAM usage: {} MB'.format(rusage.ru_maxrss/1024))

if __name__=="__main__":
    dfile = 'dsetAZ.h5'
    savedir = 'AZ'

    layer = ([128, 64, 32, 16, 8], ['elu', 'relu', 'elu', 'relu', 'elu'])
    nEpochs = 100

    DL(layer, 128, nEpochs, dfile, savedir, 0, False)
