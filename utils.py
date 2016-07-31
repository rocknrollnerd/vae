from sklearn.datasets import fetch_mldata
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def get_mnist(n=None):
    mnist = fetch_mldata('MNIST original')
    # prepare data
    data = np.float32(mnist.data[:]) / 255.
    if n is not None:
        idx = np.random.choice(data.shape[0], n, replace=False)
        return data[idx, ...]
    else:
        return data


def sample_reconstructions(test, recon, target):
    subset = np.random.randint(0, len(test), size=25)
    x = np.array(test)[np.array(subset)]
    x_recon = recon.eval({target: x})

    for i, (title, arr) in enumerate((('Originals', x), ('Reconstructions', x_recon)), 1):
        plt.subplot(1, 2, i)
        plt.title(title)
        plt.xticks(())
        plt.yticks(())
        img = np.zeros((28 * 5, 28 * 5))
        for j in xrange(5):
            for k in xrange(5):
                img[j * 28: (j + 1) * 28, k * 28: (k + 1) * 28] = arr[j * 5 + k].reshape(28, 28)
        plt.imshow(img, cmap='gray')
    plt.show()


def plot_samples(decode):
    x = np.linspace(0.1, 0.9, 20)
    v = norm.ppf(x)
    z = np.zeros((20**2, 2))
    i = 0
    for a in v:
        for b in v:
            z[i, 0] = a
            z[i, 1] = b
            i += 1
    samples = decode(z.astype('float32'))

    idx = 0
    canvas = np.zeros((28 * 20, 20 * 28))
    for i in range(20):
        for j in range(20):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = samples[idx].reshape((28, 28))
            idx += 1
    plt.title('Exploring the latent space')
    plt.imshow(canvas, cmap='gray')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.show()
