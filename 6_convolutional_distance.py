import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from numpy import random as rnd
from load import rand_pairs_c80, rand_pairs_t2k, get_pair
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

def fpdist12(x, y, w_conv, w_lr, b_lr):
    fpx_list = [fingerprint(transpose(x, i), w_conv) for i in range(12)]
    fpy = fingerprint(y, w_conv)
    fpdiff = [T.flatten((fpx[i]-fpy)**2) for fpx in fpx_list]
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
nfilters = 16
k1, k2 = (1, 12)
w_shp = (nfilters, 1, k1, k2)
w_rnd = rnd.randn(*w_shp) * init_scale / np.sqrt(nfilters * k1 * k2)
W_conv = theano.shared(floatX(w_rnd), name='W_conv')

w_rnd = rnd.randn()
b_rnd = rnd.randn()
W_lr = theano.shared(floatX(w_rnd), name='W_lr')
B_lr = theano.shared(floatX(b_rnd), name='B_lr')

d_pred = fpdist(X, Y, W_conv, W_lr, B_lr)
params = [W_conv, W_lr, B_lr]    # params = [W_conv, B_lr] ?
# d_pred = baselinedist(X, Y, W_lr, B_lr)
# params = [W_lr, B_lr]

cost = (d_pred - d_true)**2
updates = RMSprop(cost, params, lr=.01)     # default learning rate above (.001) seems to assume minitbatch size ~ 100

train = theano.function(inputs=[X, Y, d_true], outputs=cost, updates=updates)    # allow_input_downcast=True)
predict = theano.function(inputs=[X, Y], outputs=d_pred)    # allow_input_downcast=True)
share = theano.function(inputs=[], outputs=params)

n_ep = 100
# n_songs = 160                 # comment for t2k experiment
n_songs = 400                 # comment for c80 experiment
n_train = int(.75*n_songs)
training_set = range(n_train)
test_set = range(n_train, n_songs)

true_log = np.zeros((n_ep, n_train))
cost_log = np.zeros((n_ep, n_train))
pred_log = np.zeros((n_ep, n_train))
perf_log = np.zeros((n_ep, n_train))
test_log = np.zeros((n_ep, n_songs-n_train))
for ep in range(n_ep):
    print 'epoque: ' + str(ep) + '...'
    # rand_pairs = rand_pairs_c80()               # comment for t2k experiment
    rand_pairs = rand_pairs_t2k()               # comment for c80 experiment
    for s in training_set:
        X_train, Y_train, d_train = get_pair(s, **rand_pairs)
        true_log[ep, s] = d_train
        pred_log[ep, s] = predict(X_train, Y_train)
        perf_log[ep, s] = (pred_log[ep, s] > 0.5) == d_train
        cost_log[ep, s] = train(X_train, Y_train, d_train)
        # print 'training song pair: ' + str(s)
        # print X_train.shape, Y_train.shape, d_train
        # print W_lr.get_value(), B_lr.get_value()
    for s in test_set:
        X_test, Y_test, d_test = get_pair(s, **rand_pairs)
        test_log[ep, s-n_train] = (predict(X_test, Y_test) > 0.5) == d_test

        ''' PRINT STATEMENTS '''

        # print 'test song pair: ' + str(s)
    # print true_log[ep]
    # print pred_log[ep]
    # print cost_log[ep]s
    # print perf_log[ep]
    # print test_log[ep]
    # print 'mean true: ' + str(np.mean(true_log[ep]))
    # print 'mean pred: ' + str(np.mean(pred_log[ep]))
    print 'mean cost: ' + str(np.mean(cost_log[ep]))
    print 'training accuracy: ' + str(np.mean(perf_log[ep]))
    print 'test accuracy:     ' + str(np.mean(test_log[ep])) + '  (' + str(int(np.sum(test_log[ep]))) + '/' + str(len(test_log[ep])) + ')'
# print true_log
# print pred_log
# print cost_log
# print perf_log
print share()