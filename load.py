
import numpy as np
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


def coverpairs(ind):
    filelist = '/Users/Jan/Documents/Work/Cogitch/Audio/covers80/'
    filedir = '/Users/Jan/Documents/Work/Cogitch/Audio/covers80/'
    ext = '_vamp_vamp-hpcp-mtg_MTG-HPCP_HPCP.csv'

    Xfile = readfilelist(filelist + 'list1.list')
    Yfile = readfilelist(filelist + 'list2.list')

    n = len(Xfile)
    i, j, d = covers_and_noncover_pairs(n, ind)

    X = pd.read_csv(filedir + Xfile[i][0] + ext, delimiter=',').values[:,1:]
    Y = pd.read_csv(filedir + Yfile[j][0] + ext, delimiter=',').values[:,1:]

    X.reshape((1,1,-1,12))
    Y.reshape((1,1,-1,12))

    return X, Y, d


def readfilelist(filelist):
    filelistfile = open(filelist)
    listreader = csv.reader(filelistfile)
    return list(listreader)


def covers_and_noncover_pairs(n, ind):
    indX = np.repeat(range(n),2)
    d = np.tile([0, 1], n)
    indY = np.mod(indX + d, n)
    # print d.shape
    return indX[ind], indY[ind], d[ind]


def test_covers80():
    x, y, d = coverpairs(4)
    print 'testing c80'
    print type(x), type(y), type(d)
    print x.shape, y.shape, d

# test_covers80()
