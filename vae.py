import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
import numpy as np

rnd = RandomStreams()


def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - T.log(T.abs_(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma / 2. - (x - mu) ** 2 / (2. * T.exp(logsigma))


class SampleLayer(lasagne.layers.MergeLayer):
    """
    Basically taking the approach from [parmesan](https://github.com/casperkaae/parmesan/blob/master/parmesan/layers/sample.py)
    """

    def __init__(self, mean, log_sigma):
        # treating mean + logsigma as one input parameter;
        # same goes for `get_output_for`
        super(SampleLayer, self).__init__([mean, log_sigma])

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def get_output_for(self, input, **kwargs):
        mu, log_sigma = input
        epsilon = rnd.normal(mu.shape)
        return mu + T.log(1. + T.exp(log_sigma)) * epsilon


def vae_log_likelihood(z, z_mean, z_log_sigma, recon, target):
    """
    VAE loss consists of two parts:
     * encoding loss (how succesfully we've transformed the input into latent variable)
     * reconstruction loss (how good is the reconstruction)
    """
    # 1) reconstruction loss: this is just simple log-likelihood
    # assuming gaussian likelihood here (i.e., mean square error loss)
    reconstruction_loss = log_gaussian(recon, target, 1.).sum(axis=1)

    # 2) encoding loss: this is KL-divergence
    encoding_loss = log_gaussian_logsigma(z, z_mean, z_log_sigma).sum(axis=1)
    # 2.5) there's also prior
    prior = log_gaussian(z, 0., 1.).sum(axis=1)
    return T.sum(prior + reconstruction_loss - encoding_loss)
