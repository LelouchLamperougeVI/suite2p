import numpy as np
from suite2p.blat.utils import fast_smooth, accumarray
from sklearn import metrics
from scipy.ndimage import gaussian_filter
from scipy import stats

def crossvalidate(x: np.ndarray, n: np.ndarray, bins=80, dt=15, sigma=2, k=10):
    n = n * 100
    cv = np.floor(x.shape[0] / k)
    cv = np.repeat(np.arange(k), cv)
    cv = np.concatenate((cv, [k] * (x.shape[0] - cv.shape[0])))

    real = np.digitize(x, bins=np.linspace(0, np.max(x), bins+1))
    decoded = np.zeros_like(x)
    likelihood = np.zeros((x.shape[0], bins))
    for i in range(k):
        decoded[cv == i], likelihood[cv == i, :] = decode(x[cv != i], n[:, cv != i], n[:, cv == i], bins=bins, dt=dt, sigma=sigma)

    cm = metrics.confusion_matrix(real, decoded)
    cm = gaussian_filter(cm / np.mean(cm), sigma)

    idx = real
    real = real * np.max(x) / bins
    decoded = decoded * np.max(x) / bins

    error = np.array([np.abs(real - decoded), np.max(x) - np.abs(real - decoded)])
    error = np.min(error, axis=0)
    mu = accumarray(idx, error, func=np.mean)
    sem = accumarray(idx, error, func=stats.sem)
    
    error = np.sqrt(np.mean((real - decoded)**2))

    ret = {
        'real': real,
        'decoded': decoded,
        'likelihood': likelihood,
        'cm': cm,
        'error': {
            'overall': error,
            'error': mu,
            'sem': sem,
        },
    }

    return ret

def decode(x: np.ndarray, train: np.ndarray, test: np.ndarray, bins=80, dt=15, sigma=2, penalty=1e-2):
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
    log_fx[np.isinf(log_fx) | np.isnan(log_fx)] = penalty
    precomp = dt * np.sum(fx, axis=1) + np.log(occ)
    likelihood = np.array([np.sum(spks * log_fx, axis=1) - precomp for spks in n.T])
    decoded = np.argmax(likelihood, axis=1)

    return decoded, likelihood