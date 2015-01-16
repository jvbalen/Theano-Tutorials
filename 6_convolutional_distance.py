import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from numpy import random as rnd
from load import get_rand_pairs, get_pair
from theano.tensor.nnet.conv import conv2d
# from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def logistic(x, w, b=0):
    wx = T.dot(x, w) + b
    return 1/(1+T.exp(-wx))


def sigmoid(x, w, b=0):
    wx = w * T.sqrt(T.sum(x**2)) + b
    return 1/(1+T.exp(-wx))


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def fingerprint(x, w_conv):
    x_conv = conv2d(x, w_conv)
    return T.dot(x_conv.dimshuffle((0, 1, 3, 2)), x_conv)


def fpdist(x, y, w_conv, w_lr, b_lr):
    fpx = fingerprint(x, w_conv)
    fpy = fingerprint(y, w_conv)
    fpdiff = T.flatten((fpx-fpy)**2)
    fpdiff = fpdiff/T.sum(fpdiff)
    return sigmoid(fpdiff, w_lr, b_lr)


def baselinedist(x, y, w_lr, b_lr):
    fpx = T.dot(x.dimshuffle((0, 1, 3, 2)), x)
    fpy = T.dot(y.dimshuffle((0, 1, 3, 2)), y)
    fpdiff = T.flatten((fpx-fpy)**2)
    fpdiff = fpdiff/T.sum(fpdiff)
    return sigmoid(fpdiff, w_lr, b_lr)


X = T.tensor4(name='X')
Y = T.tensor4(name='Y')
d_true = T.scalar()

init_scale = 1
nfilters = 10
k1, k2 = (1, 12)
w_shp = (nfilters, 1, k1, k2)
w_rnd = rnd.randn(*w_shp) * init_scale / np.sqrt(nfilters * k1 * k2)
W_conv = theano.shared(floatX(w_rnd), name='W_conv')

w_rnd = rnd.randn()
b_rnd = rnd.randn()
W_lr = theano.shared(floatX(w_rnd), name='W_lr')
B_lr = theano.shared(floatX(b_rnd), name='B_lr')

# d_pred = fpdist(X, Y, W_conv, W_lr, B_lr)
# params = [W_conv, W_lr, B_lr]
d_pred = baselinedist(X, Y, W_lr, B_lr)
params = [W_lr, B_lr]

cost = (d_pred - d_true)**2
updates = RMSprop(cost, params, lr=0.1)     # default learning rate above (.001) seems to asume minitbatch size ~ 100

train = theano.function(inputs=[X, Y, d_true], outputs=cost, updates=updates)    # allow_input_downcast=True)
predict = theano.function(inputs=[X, Y], outputs=d_pred)    # allow_input_downcast=True)
share = theano.function(inputs=[], outputs=params)

n_ep = 100
n_songs = 160
true_log = np.zeros((n_ep, n_songs))
cost_log = np.zeros((n_ep, n_songs))
pred_log = np.zeros((n_ep, n_songs))
perf_log = np.zeros((n_ep, n_songs))
for ep in range(n_ep):
    print 'epoque: ' + str(ep) + '...'
    rand_pairs = get_rand_pairs()
    for s in range(n_songs):
        # print 'song pair: ' + str(s)
        X_train, Y_train, d_train = get_pair(s, **rand_pairs)
        # print X_train.shape, Y_train.shape
        true_log[ep, s] = d_train
        cost_log[ep, s] = train(X_train, Y_train, d_train)
        pred_log[ep, s] = predict(X_train, Y_train)
        perf_log[ep, s] = (pred_log[ep, s] > 0.5) == d_train
    print true_log[ep]
    print pred_log[ep]
    # print cost_log[ep]
    # print perf_log[ep]
    print 'mean true: ' + str(np.mean(true_log[ep]))
    print 'mean pred: ' + str(np.mean(pred_log[ep]))
    print 'mean cost: ' + str(np.mean(cost_log[ep]))
    print 'mean perf: ' + str(np.mean(perf_log[ep]))
    # print share()
# print true_log
# print pred_log
# print cost_log
# print perf_log
print share()