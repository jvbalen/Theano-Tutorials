import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from numpy import random as rnd
from load import coverpairs
from theano.tensor.nnet.conv import conv2d
# from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


def logistic(x, w):
    wx = T.dot(x, w)
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
    return T.dot(x_conv, x.T)


def fpdist(x, y, w_conv, w_lr):
    fpx = fingerprint(x, w_conv)
    fpy = fingerprint(y, w_conv)
    fpdiff = T.flatten((fpx-fpy)**2)
    return logistic(fpdiff, w_lr)    # previously: T.nnet.softmax(T.dot(fpdiff, W-o)) but that doesn't work for scalars


X = T.ftensor4(name='X')
Y = T.ftensor4(name='Y')
d_true = T.scalar()

nfilters = 32
k1, k2 = (24, 1)
w_shp = (nfilters, 1, k1, k2)
w_rnd = rnd.random_sample(w_shp)
W_conv = theano.shared(np.asarray(w_rnd, dtype=X.dtype), name='W_conv')

nbins = 12
w_rnd = rnd.random_sample((nbins*nbins,))
W_lr = theano.shared(np.asarray(w_rnd, dtype=X.dtype), name='W_lr')

d_pred = fpdist(X, Y, W_conv, W_lr)
cost = (d_pred - d_true)**2
params = [W_conv, W_lr]

updates = RMSprop(cost, params, lr=0.001)
train = theano.function(inputs=[X, Y, d_true], outputs=cost, updates=updates)    # allow_input_downcast=True)
predict = theano.function(inputs=[X, Y], outputs=d_pred)    # allow_input_downcast=True)


n_ep = 100
n_songs = 80
cost_array = np.zeros((n_ep, n_songs))
for ep in range(n_ep):
    print 'epoque: ' + str(ep)
    for s in range(n_songs):
        X_train, Y_train, d_train = coverpairs(s)
        X_train.reshape((1, -1, 12))
        Y_train.reshape((1, -1, 12))
        # print type(x), type(y)
        # print x.shape, y.shape
        # print fpdist(x, y, W, W_lr), d
        cost_array[ep, s] = train(X_train, Y_train, d_train)
    print 'mean cost: ' + str(np.mean(cost_array[ep]))


# Old stuff

# srng = RandomStreams()

# def floatX(X):
#     return np.asarray(X, dtype=theano.config.floatX)
#
# def init_weights(shape):
# #     return theano.shared(np.random.randn(*shape) * 0.01)
#     return theano.shared(floatX(np.random.randn(*shape) * 0.01))
#
# def rectify(X):
#     return T.maximum(X, 0.)
#
# def softmax(X):
#     e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
#     return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
#
# def dropout(X, p=0.):
#     if p > 0:
#         retain_prob = 1 - p
#         X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
#         X /= retain_prob
#     return X


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