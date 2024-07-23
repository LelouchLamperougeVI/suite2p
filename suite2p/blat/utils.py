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
    if sigma == 0:
        return A
        
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

def corr(X, Y=None, axis=1):
    """
    I can't believe python doesn't have a proper function for getting Pearson correlations...
    """
    if Y is None:
        Y = X.copy()
    if (X.ndim > 2) | (Y.ndim > 2):
        raise ValueError('X and Y must be either vectors or 2D matrices.')
    if axis not in (0, 1):
        raise ValueError('axis must be either 0 or 1.')

    if X.ndim == 1:
        X = np.array([X])
        if axis == 0:
            X = X.T
    if Y.ndim == 1:
        Y = np.array([Y])
        if axis == 0:
            Y = Y.T
    if axis == 1:
        X, Y = X.T, Y.T

    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    rho = X.T @ Y / np.sqrt(np.sum(X**2, axis=0)[:, np.newaxis] @ np.sum(Y**2, axis=0)[np.newaxis, :])

    rho[np.isnan(rho)] = 0
    return rho
        

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


from itertools import product
import numpy as np


def accumarray(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.
    Stolen from https://scipy.github.io/old-wiki/pages/Cookbook/AccumarrayLike.html

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array. 
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out
    
