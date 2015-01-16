
import numpy as np
from numpy import random as rnd
import pandas as pd
import csv
import os


datasets_dir = '/media/datasets/'


def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h


def mnist(ntrain=60000,ntest=10000,onehot=True):
    data_dir = os.path.join(datasets_dir,'mnist/')
    fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 *28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 *28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX/ 255.
    teX = teX / 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


def get_pair(ind, filedir, ext, xlist, ylist, i, j, d):

    i, j, d = i[ind], j[ind], d[ind]
    x = pd.read_csv(filedir + xlist[i][0] + ext, delimiter=',').values[:,1:]
    y = pd.read_csv(filedir + ylist[j][0] + ext, delimiter=',').values[:,1:]

    return x.reshape((1, 1, -1, 12)), y.reshape((1, 1, -1, 12)), d


def readfilelist(filelist):
    filelistfile = open(filelist)
    listreader = csv.reader(filelistfile)
    return list(listreader)


def get_rand_pairs():
    """
    :param n: number of true cover pairs in collection. total number of pairs returned (true + false) will be 2n.
    :return: filedir, ext, xlist, ylist
        i = index of first song, j = index of second song, d = distance (= 0 for true pairs, 1 for false pairs)
    """

    filelist = '/Users/Jan/Documents/Work/Cogitch/Audio/covers80/'
    filedir = '/Users/Jan/Documents/Work/Cogitch/Audio/covers80/'
    ext = '_vamp_vamp-hpcp-mtg_MTG-HPCP_HPCP.csv'

    xlist = readfilelist(filelist + 'list1.list')
    ylist = readfilelist(filelist + 'list2.list')

    n = len(xlist)
    d = np.tile([0, 1], n)
    indX = np.repeat(range(n), 2)
    indY = np.mod(indX + d, n)

    randperm = rnd.permutation(2 * n)
    pairs = {
        'filedir': filedir,
        'ext': ext,
        'xlist': xlist,
        'ylist': ylist,
        'i': indX[randperm],
        'j': indY[randperm],
        'd': d[randperm]}
    return pairs


# def test_covers80():
#     x, y, d = coverpairs(4)
#     print 'testing c80'
#     print type(x), type(y), type(d)
#     print x.shape, y.shape, d

# test_covers80()
