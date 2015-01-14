__author__ = 'Jan'

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import coverpairs
from theano.tensor.nnet import conv2d
# from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

x = T.ftensor4
y = T.ftensor4

nfilters = 32
filtershape = (24, 1)
w_shape = (nfilters, 1) + filtershape
w = theano.shared(np.random.randn(*w_shape))

def fpdist(x, y, w):
    x_conv = conv2d(x, w)
    y_conv = conv2d(y, w)
    fpx = T.reshape(T.dot(x_conv, x.T), (-1,))
    fpy = T.reshape(T.dot(y_conv, y.T), (-1,))
    return T.nnet.categorical_crossentropy(fpx, fpy)

d_out = fpdist(x, y, w)
d_true = T.vector()
cost = (d_out - d_true)**2
params = [w]

updates = RMSprop(cost, params, lr=0.001)
train = theano.function(inputs=[x, y, d_true], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[x, y], outputs=d_out, allow_input_downcast=True)


# Revise from here


# def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
#     l1a = rectify(conv2d(X, w, border_mode='full'))
#     l1 = max_pool_2d(l1a, (2, 2))
#     l1 = dropout(l1, p_drop_conv)
#
#     l2a = rectify(conv2d(l1, w2))
#     l2 = max_pool_2d(l2a, (2, 2))
#     l2 = dropout(l2, p_drop_conv)
#
#     l3a = rectify(conv2d(l2, w3))
#     l3b = max_pool_2d(l3a, (2, 2))
#     l3 = T.flatten(l3b, outdim=2)
#     l3 = dropout(l3, p_drop_conv)
#
#     l4 = rectify(T.dot(l3, w4))
#     l4 = dropout(l4, p_drop_hidden)
#
#     pyx = softmax(T.dot(l4, w_o))
#     return l1, l2, l3, l4, pyx
#
# trX, teX, trY, teY = mnist(onehot=True)
#
# trX = trX.reshape(-1, 1, 28, 28)
# teX = teX.reshape(-1, 1, 28, 28)
#
# X = T.ftensor4()
# Y = T.fmatrix()
#
# w = init_weights((32, 1, 3, 3))
# w2 = init_weights((64, 32, 3, 3))
# w3 = init_weights((128, 64, 3, 3))
# w4 = init_weights((128 * 3 * 3, 625))
# w_o = init_weights((625, 10))
#
# noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
# l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
# y_x = T.argmax(py_x, axis=1)
#
#
# cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
# params = [w, w2, w3, w4, w_o]
# updates = RMSprop(cost, params, lr=0.001)
#
# train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
# predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
#
# for i in range(100):
#     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
#         cost = train(trX[start:end], trY[start:end])
#     print np.mean(np.argmax(teY, axis=1) == predict(teX))

# X = [x.reshape((1, 1, X.shape[0], X.shape[1])) for x in X]
# Y = [y.reshape((1, 1, Y.shape[0], Y.shape[1])) for y in Y]

n_ep = 100
n_songs = 80
cost = np.zeros((n_ep, n_songs))
for ep in range(n_ep):
    print 'epoque: ' + str(ep)
    for it in range(n_songs):
        X, Y, D = coverpairs(it)
        X.reshape((1, -1, 12))
        Y.reshape((1, -1, 12))
        print type(X), type(Y)
        print X.shape, Y.shape
        cost[ep, it] = train(X, Y, D)
    print 'mean cost: ' + str(np.mean(cost[ep]))
