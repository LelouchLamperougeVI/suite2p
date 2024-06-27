"""
KSG estimator
https://doi.org/10.1103/PhysRevE.69.066138
"""

import numpy as np
from scipy.special import digamma
from scipy.stats import zscore, rankdata
from scipy.spatial import KDTree
from joblib import Parallel, delayed
import warnings

def ksg_mi(x: np.ndarray, y: np.ndarray, k=5, method=1) -> np.ndarray:
    """
    KSG estimator for mutual information.
    This is an implementation of the second algorithm
    as we cannot make assumptions on equality of marginal
    spaces, especially with calcium data.
    Units of bits.
    I did everything to extract every bit of performance T_T
    """
    if x.ndim != 2:
        raise ValueError('x needs to be a 2-dimensional array of shape (n_features, n_samples)')
    if y.ndim != 1:
        raise ValueError('y needs to be a vector')

    if x.dtype != np.dtype(np.float64): # for speed
        x = x.astype(np.float64)
    if y.dtype != np.dtype(np.float64):
        y = y.astype(np.float64)

    def job(i):
        joint = np.array([x[i, :], y]).T
        joint = twocol_unique(joint)
        joint = zscore(joint, axis=0)
        
        argx = np.argsort(joint[:, 0])
        sortx = joint[argx, 0]
        argy = np.argsort(joint[:, 1])
        sorty = joint[argy, 1]
        
        tree = KDTree(joint)
        _, neighbours = tree.query(joint, k=k)
        e = np.array([joint - joint[neigh, :] for neigh in neighbours.T])
        e = np.max(np.abs(e), axis=0)
    
        n = np.zeros_like(joint)
        n[:, 0] = np.searchsorted(sortx, sortx + e[argx, 0], side='right') - \
                    np.searchsorted(sortx, sortx - e[argx, 0], side='left')
        n[:, 1] = np.searchsorted(sorty, sorty + e[argy, 1], side='right') - \
                    np.searchsorted(sorty, sorty - e[argy, 1], side='left')

        I = digamma(k) - 1/k - np.mean(np.sum(digamma(n), axis=1)) + digamma(n.shape[0])
        return I

    I = np.array(Parallel(n_jobs=-1, backend='threading')(delayed(job)(i) for i in range(x.shape[0])))
        
    if np.any(I < 0):
        warnings.warn('Estimated mutual information contains negatives.', RuntimeWarning)
    
    return I * np.log2(np.exp(1))


def twocol_unique(x: np.ndarray) -> np.ndarray:
    """
    Hacky way to get unique rows in a two columns array,
    FAST!!!!!
    """
    ranks = np.apply_along_axis(rankdata, 0, x).astype('uint32')
    ranks = np.ascontiguousarray(ranks)
    vect = ranks.view(np.uint64)
    _, idx = np.unique(vect[:, 0], return_index=True)

    return x[idx, :]
