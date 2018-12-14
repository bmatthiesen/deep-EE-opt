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
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
import dl as dl

import argparse

parser = argparse.ArgumentParser(description='Calcuate validation loss')
parser.add_argument('model')
parser.add_argument('outfile')
parser.add_argument('key')
parser.add_argument('index', type=int)
args = parser.parse_args()

rf = '../../data/dset_final.h5'

with h5py.File(rf,'r') as f:
    inp = f['validation/input'][...]
    obj = f['validation/objval'][...]

def predict(model, inp):
    pred = np.asarray(model.predict(inp), dtype=np.float)
    predlin = K.clip(tf.Variable(10**pred), 0, 1)
    Pmax = 10**np.array(inp[:,-1])
    obj = K.eval(dl.calcObjective([tf.Variable(10**np.asarray(inp, dtype=np.float)), predlin]))

    return obj


model = keras.models.load_model(args.model)
pred = predict(model, inp)
loss = np.mean((obj-pred)**2)

with h5py.File(args.outfile, 'r+') as f:
    f[args.key][args.index] = loss
