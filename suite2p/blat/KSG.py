"""
KSG estimator
https://doi.org/10.1103/PhysRevE.69.066138
"""

import numpy as np
from scipy.special import digamma
from scipy.stats import zscore
from scipy.spatial import KDTree
from joblib import Parallel, delayed
import numba as nb
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
import warnings

def ksg_mi(x: np.ndarray, y: np.ndarray, k=5) -> np.ndarray:
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

    def job(i):
        joint = np.array([x[i, :], y]).T
        joint, _, _ = nb_unique(joint, axis=0) # sparsify marginals to ensure continuity in Laplacian
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
        n[:, 0] = np.searchsorted(sortx, sortx[argx] + e[:, 0], side='right') - \
                    np.searchsorted(sortx, sortx[argx] - e[:, 0], side='left')
        n[:, 1] = np.searchsorted(sorty, sorty[argy] + e[:, 1], side='right') - \
                    np.searchsorted(sorty, sorty[argy] - e[:, 1], side='left')

        I = digamma(k) - 1/k - np.mean(np.sum(digamma(n), axis=1)) + digamma(n.shape[0])
        return I

    I = np.array(Parallel(n_jobs=-1, prefer='threads')(delayed(job)(i) for i in range(x.shape[0])))
        
    if np.any(I < 0):
        warnings.warn('Estimated mutual information contains negatives.', RuntimeWarning)
    
    return I * np.log2(np.exp(1))

"""
Code below was stolen from rishi-kulkarni
https://github.com/numba/numba/issues/7663
"""

@overload(np.all)
def np_all(x, axis=None):
    # ndarray.all with axis arguments for 2D arrays.
    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError("Only accepts NumPy ndarray")

    if not (1 <= x.ndim <= 2):
        raise TypingError("Only supports 1D or 2D NumPy ndarrays")

    if isinstance(axis, types.NoneType):
        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)
        return _np_all_impl

    elif x.ndim == 1:
        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)
        return _np_all_impl

    elif x.ndim == 2:
        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)
        return _np_all_impl

    else:
        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)
        return _np_all_impl

@nb.jit(nopython=True, cache=True)
def nb_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True)
    
    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0
    Returns
    -------
    2D array
        unique rows (or columns) from the input array
    1D array of ints
        indices of unique rows (or columns) in input array
    1D array of ints
        number of instances of each unique row
    """
    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()
    else:
        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind="mergesort")
        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]
    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1
    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx], orig_idx[idx], counts
