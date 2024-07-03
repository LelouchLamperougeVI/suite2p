import numpy as np
from suite2p.blat.utils import fast_smooth
from sklearn import metrics
from scipy.ndimage import gaussian_filter

def crossvalidate(x: np.ndarray, n: np.ndarray, bins=80, dt=15, sigma=2, k=10):
    n = n * 100
    cv = np.floor(x.shape[0] / k)
    cv = np.repeat(np.arange(k), cv)
    cv = np.concatenate((cv, [k] * (x.shape[0] - cv.shape[0])))

    real = np.digitize(x, bins=np.linspace(0, np.max(x), bins+1))
    decoded = np.zeros_like(x)
    for i in range(k):
        decoded[cv == i] = decode(x[cv != i], n[:, cv != i], n[:, cv == i], bins=bins, dt=dt, sigma=sigma)

    cm = metrics.confusion_matrix(real, decoded)
    cm = gaussian_filter(cm / np.mean(cm), sigma)

    real = real * np.max(x) / bins
    decoded = decoded * np.max(x) / bins
    error = np.sqrt(np.mean((real - decoded)**2))

    ret = {
        'real': real,
        'decoded': decoded,
        'cm': cm,
        'error': error,
    }

    return ret

def decode(x: np.ndarray, train: np.ndarray, test: np.ndarray, bins=80, dt=15, sigma=2):
    ranges = ([0, np.max(x)],)
    occ, _ = np.histogramdd(x, range=ranges, bins=bins)
    fx = np.array([np.histogramdd(x, range=ranges, bins=bins, weights=train[i, :])[0] for i in range(train.shape[0])])
    fx = fx / occ
    occ = occ / np.sum(occ)
    fx = fast_smooth(fx, sigma=sigma, axis=1)

    kernel = np.ones((dt,))
    normalizer = np.convolve(np.ones((test.shape[1],)), kernel/dt, mode='same')
    n = np.array([np.convolve(spks, kernel, 'same') for spks in test])
    n = n / normalizer

    fx = fx.T
    log_fx = np.log(fx)
    log_fx[np.isinf(log_fx)] = 1e-2
    precomp = dt * np.sum(fx, axis=1) + np.log(occ)
    decoded = [np.argmax(np.sum(spks * log_fx, axis=1) - precomp) for spks in n.T]

    return decoded
