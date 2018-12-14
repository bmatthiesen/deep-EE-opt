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
import sys
import platform
import datetime
import time
import itertools as it
import HataCOST231
import fading

num_UE = 4
num_BS = 4
num_BS_Ant = 2

PLE = 4.5
d0 = 35
f0 = 1800e6
PL0 = 10**(-8.4)

B = 180e3
N0 = 1e-3 * 10**(-174/10)
F = 10**(3/10)
noise = B*N0*F

pos_BS = np.array([[500,500], [-500,500], [500,-500], [-500,-500]])

def crandn(*args, **kwargs):
    return 1/np.sqrt(2) * (np.random.randn(*args, **kwargs) + 1j * np.random.randn(*args, **kwargs))

def PL_Alessio(dist):
    return 2*PL0 / (1+(dist / d0)**PLE)

def HataSuburban(dist):
    pl = HataCOST231.suburban(dist/1000, 1900, 30, 1.5) # in dB
    pl += fading.shadowing(8, dist.shape)
    return 10**(-pl/10)

def HataSuburban_noSF(dist):
    pl = HataCOST231.suburban(dist/1000, 1900, 30, 1.5) # in dB
    return 10**(-pl/10)

def HataUrban_noSF(dist):
    pl = HataCOST231.urban(dist/1000, 1900, 30, 1.5, metropolitan = False) # in dB
    return 10**(-pl/10)

def HataUrban(dist):
    pl = HataCOST231.urban(dist/1000, 1900, 30, 1.5, metropolitan = False) # in dB
    pl += fading.shadowing(8, dist.shape)
    return 10**(-pl/10)

def defDist():
    pos_UE = np.random.uniform(-500, 500, (num_UE,2)) + pos_BS

    dist = np.zeros((num_UE,num_BS,2))
    for i in range(dist.shape[0]):
        dist[i] = pos_UE[i] - pos_BS
    dist = np.linalg.norm(dist,2,axis=2)

    return dist

def create_channel(PL = PL_Alessio, DIST = defDist):
    dist = DIST()

    slow_fading = np.sqrt(PL(dist))
    fast_fading = crandn(num_UE,num_BS,num_BS_Ant)

    channel = np.zeros(fast_fading.shape, dtype = complex)

    for i in range(fast_fading.shape[2]):
        channel[:,:,i] = fast_fading[:,:,i] * slow_fading

    # create matched filter for MRT
    mrt = np.diagonal(channel)
    mrt = (mrt/np.linalg.norm(mrt,axis=0)).T

    # effective channel
    eff = np.sum(np.swapaxes(channel,0,1) * np.conj(mrt), axis=-1).T

    # effective channel gain (including noise)
    MF = np.abs(eff)**2 / noise

    """" This does the same but is slower:
    MF = np.empty(channel.shape[:2])
    for k, i in it.product(range(channel.shape[0]), range(channel.shape[1])):
        MF[k,i] = np.abs(np.vdot(channel[k,k], channel[k,i]))**2 / np.linalg.norm(channel[k,k])**2 / noise
    """

    return (MF, channel)


def check_channel(mf):
    return np.all(np.argmax(mf,axis=1) == range(mf.shape[1]))


def gen_channel(*args, **kwargs):
    while (True):
        (mf, chan) = create_channel(*args, **kwargs)

        if check_channel(mf):
            break

    return (mf, chan)


def make(fn, numChans, chunksize = 1000, **kwargs):
    with h5py.File(fn, 'w-') as f:
        # store some info
        g = f.create_group('channel_generation')

        # save source code
        g.create_dataset('source code', (1,), dtype = h5py.special_dtype(vlen=str))
        with open(__file__, "r", encoding='utf') as src:
            g['source code'][:] = "".join(src.readlines())

        # save relevant versions
        g.create_dataset('python version', (1,), dtype = h5py.special_dtype(vlen=str))
        g['python version'][:] = sys.version

        g.create_dataset('numpy version', (1,), dtype = h5py.special_dtype(vlen=str))
        g['numpy version'][:] = np.version.version

        g.create_dataset('platform', (1,), dtype = h5py.special_dtype(vlen=str))
        g['platform'][:] = platform.platform()

        # save random state
        state_dt = np.dtype([('before channel idx',np.uint64), ('state', [('keys', np.uint32, 624), ('pos', np.int), ('has_gauss', np.int), ('cached_gaussian', np.float)])])
        g.create_dataset('MT19937 state', dtype = state_dt, shape = (2,), maxshape = (None, ), chunks = (1, ))
        
        state = np.random.get_state()
        assert(state[0] == 'MT19937')
        g['MT19937 state'][0] = np.asarray((0,state[1:]),dtype=state_dt)

        # generate the channels
        g2 = f.create_group('input')
        dset_mf = g2.create_dataset('channel_to_noise_matched', (numChans, num_UE, num_BS), dtype=np.float32, maxshape = (None, num_UE, num_BS), chunks = (1, num_UE, num_BS))
        dset_chan = g2.create_dataset('channel', (numChans, num_UE, num_BS, num_BS_Ant), dtype='c8', maxshape = (None, num_UE, num_BS, num_BS_Ant), chunks = (1, num_UE, num_BS, num_BS_Ant))

        # preallocate
        cache_mf = np.empty((chunksize, num_UE, num_BS))
        cache_chan = np.empty((chunksize, num_UE, num_BS, num_BS_Ant), dtype = complex)
        idx = 0

        # go for it!
        tic = time.process_time()
        for i in range(numChans):
            # output
            if i%1000 == 0:
                perc_done = i/numChans

                try:
                    toc = time.process_time()
                    eta = (toc-tic) * (1/perc_done - 1)
                except ZeroDivisionError:
                    eta = 0

                print('{:{width},} / {:,} ({:6.2%}): ETA {}'.format(i, numChans, perc_done, datetime.timedelta(seconds=round(eta)), width=len('{:,}'.format(numChans))))

            # get channel
            (cache_mf[idx], cache_chan[idx]) = gen_channel(**kwargs)
            idx += 1

            # save chunk to HDF
            if idx % chunksize == 0:
                idx = 0
                start = int(i/chunksize)*chunksize

                dset_mf[start:start+chunksize] = cache_mf
                dset_chan[start:start+chunksize] = cache_chan
        
        # write last chunk
        if idx != 0:
            start = int(i/chunksize)*chunksize

            dset_mf[start:start+idx] = cache_mf[:idx]
            dset_chan[start:start+idx] = cache_chan[:idx]

        # save random state
        state = np.random.get_state()
        assert(state[0] == 'MT19937')
        g['MT19937 state'][1] = np.asarray((numChans+1,state[1:]),dtype=state_dt)

        # output
        print('Done. Time: {}'.format(datetime.timedelta(seconds=round(time.process_time()-tic))))



if __name__=="__main__":
    make('../../data/channels.h5', 12200)
    make('../../data/channels-hataUrban-noSF.h5', int(1e3), PL = HataUrban_noSF)
    make('../../data/channels-hataUrban.h5', int(1e3), PL = HataUrban)
