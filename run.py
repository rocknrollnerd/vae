from utils import get_mnist, sample_reconstructions, plot_samples
from vae import SampleLayer, vae_log_likelihood

import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import rectify, identity

import numpy as np
from sklearn.cross_validation import train_test_split


if __name__ == '__main__':
    data = get_mnist()

    train, test = train_test_split(data, test_size=0.1)

    _train = theano.shared(train, borrow=True)
    _test = theano.shared(test, borrow=True)

    batch_size = 500
    latent_size = 2

    target = T.matrix()

    encoder = lasagne.layers.InputLayer((None, train.shape[1]), target)
    encoder = lasagne.layers.DenseLayer(encoder, num_units=100, nonlinearity=rectify)
    mean = lasagne.layers.DenseLayer(encoder, num_units=latent_size, nonlinearity=identity)
    log_sigma = lasagne.layers.DenseLayer(encoder, num_units=latent_size, nonlinearity=identity)
    z = SampleLayer(mean=mean, log_sigma=log_sigma)
    decoder1 = lasagne.layers.DenseLayer(z, num_units=100, nonlinearity=rectify)
    decoder2 = lasagne.layers.DenseLayer(decoder1, num_units=train.shape[1], nonlinearity=rectify)
    decoder = decoder2

    z_actual = lasagne.layers.get_output(z, deterministic=False)
    z_mean = lasagne.layers.get_output(mean, deterministic=False)
    z_log_sigma = lasagne.layers.get_output(log_sigma, deterministic=False)
    recon = lasagne.layers.get_output(decoder, deterministic=False)

    ll = vae_log_likelihood(z_actual, z_mean, z_log_sigma, recon, target)
    ll /= batch_size

    params = lasagne.layers.get_all_params(decoder, trainable=True)
    updates = lasagne.updates.adam(-ll, params, learning_rate=0.001)

    i = T.iscalar()

    train_fn = theano.function(
        [i], ll, updates=updates,
        givens={
            target: _train[i * batch_size: (i + 1) * batch_size]
        })
    test_fn = theano.function(
        [i], ll,
        givens={
            target: _test[i * batch_size: (i + 1) * batch_size]
        })

    num_train_batches = train.shape[0] / batch_size
    num_test_batches = test.shape[0] / batch_size

    for e in xrange(30):
        train_errs = []
        test_errs = []
        for idx in xrange(num_train_batches):
            train_errs.append(train_fn(idx))
        for idx in xrange(num_test_batches):
            test_errs.append(test_fn(idx))
        print 'epoch', e, 'train err', np.mean(train_errs), 'test err', np.mean(test_errs)

    sample_reconstructions(test, recon, target)

    # construct separate decoder
    z_input = T.matrix()
    single_decoder = lasagne.layers.InputLayer((None, latent_size), z_input)
    single_decoder = lasagne.layers.DenseLayer(single_decoder, num_units=100, nonlinearity=rectify, W=decoder1.W, b=decoder1.b)
    single_decoder = lasagne.layers.DenseLayer(single_decoder, num_units=100, nonlinearity=rectify, W=decoder2.W, b=decoder2.b)
    decode = theano.function([z_input], lasagne.layers.get_output(single_decoder))

    plot_samples(decode)
