"""
Useful generic helper functions.
"""

import numpy as np
from scipy import signal
from joblib import Parallel, delayed

def fast_smooth(A: np.ndarray, sigma: int, axis=-1) -> np.ndarray:
    """
    Fast 1D Gaussian smoothing with edge correction.
    """
    x = np.arange(-sigma*5, sigma*5 + 1) # 5 sigma baby!
    kernel = np.exp(-.5 * x**2 / sigma**2)
    kernel = (kernel / np.sum(kernel))
    # normalizer = signal.convolve(np.ones((A.shape[axis],)), kernel, mode='same')

    def smooth(a):
        normalizer = signal.convolve((~np.isnan(a)).astype(np.float64), kernel, mode='same')
        a[np.isnan(a)] = 0
        ret = signal.convolve(a, kernel, mode='same') / normalizer
        return ret

    A = np.moveaxis(A, source=axis, destination=-1)
    og_shape = A.shape
    A = A.reshape((np.prod(A.shape[:-1]).astype(int), A.shape[-1]))
    A = np.array(Parallel(n_jobs=-1, prefer="threads")(delayed(smooth)(v) for v in A))
    # A = np.array([smooth(v) for v in A])
    A = A.reshape(og_shape)
    A = np.moveaxis(A, source=-1, destination=axis)
    
    return A

def knnsearch(target, query):
    """
    MATLAB-like knnsearch for the closest neighbour.
    Distilled version to suite my purpose...
    """
    idx = np.searchsorted(target, query)
    side = np.argmin([np.abs(target[idx] - query), np.abs(target[idx - 1] - query)], axis=0)
    idx[side == 1] -= 1
    idx[idx == target.shape[-1]] -= 1
    return idx

def fill_gaps(X, gap):
    """
    Get continuous segments in logical array by filling in small gaps.
    """
    idx = np.flatnonzero(X)
    gaps = np.flatnonzero((np.diff(idx) <= (gap + 1)) & (np.diff(idx) > 1))
    for g in gaps:
        X[idx[g]:idx[g+1]] = True

    return X
    
