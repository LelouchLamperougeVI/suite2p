"""
Useful generic helper functions.
"""

import numpy as np
from scipy import signal, stats, special
from joblib import Parallel, delayed
from itertools import product
import numpy as np


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


def gethead(x, tail=False):
    if tail:
        tails = gethead(x[::-1])
        return tails[::-1]
    heads = np.diff(x.astype(bool).astype(int))
    heads = heads > 0
    heads = np.insert(heads, 0, False)
    return heads


def corr(X, Y=None, axis=1, pval=False):
    """
    I can't believe python doesn't have a proper function for getting Pearson
    correlations... Scipy's implementation is slow as fuck. P-values are calculated
    in the same manner specified in scipy.stats.pearsonr. NaN values will be ignored.

    Inputs
    ------
    X (array):
        Input data array. Must be either vector or 2D matrix.
    Y (array - optional):
        Optional second array. If given, returns matrix of size (m, n), where m is
        the dimension of X and n is the dimension of Y.
    axis (int):
        Axis over which to computer pairwise Pearson Correlation coefficients.
    pval (bool):
        Whether to return the p-values. If True, method returns tuple (rho, pval).
        Otherwise (default False), only returns rho.

    Returns
    -------
    rho (array):
        Pearson correlation coefficients. If only X is given, conducts pairwise
        correlations between vectors specified along axis.
    (rho, pval):
        If pval == True, returns p-values.
    """
    sym = False
    if Y is None:
        Y = X.copy()
        sym = True

    if (X.ndim > 2) | (Y.ndim > 2):
        raise ValueError('X and Y must be either vectors or 2D matrices.')
    if axis not in (0, 1):
        raise ValueError('axis must be either 0 or 1.')

    if X.ndim == 1:
        X = np.array([X])
        X = X.T
    elif axis == 1:
        X = X.T
    if Y.ndim == 1:
        Y = np.array([Y])
        Y = Y.T
    elif axis == 1:
        Y = Y.T

    n = ~np.isnan(X).astype(int).T @ ~np.isnan(Y).astype(int)

    X = X - np.nanmean(X, axis=0)
    Y = Y - np.nanmean(Y, axis=0)
    X[np.isnan(X)] = 0
    Y[np.isnan(Y)] = 0
    rho = X.T @ Y / np.sqrt(np.sum(X**2, axis=0)[:, np.newaxis] @ np.sum(Y**2, axis=0)[np.newaxis, :])

    if sym:
        rho = (rho + rho.T) / 2 # guarantee symmetry

    if not pval:
        return rho

    unique_n = np.unique(n)
    p = np.zeros_like(rho)
    for u in unique_n:
        dist = stats.beta(u / 2 - 1, u / 2 - 1, loc=-1, scale=2)
        r, c = np.nonzero(n == u)
        p[r, c] = 2 * dist.cdf(-abs(rho[r, c]))

    return rho, p


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


def bino_cdf(x, n, p, verbose=False):
    """
    Calculate the CDF of a binomial distribution. For large N, uses the Normal
    approximation. When the 3-sigma rule is not satisfied (Berry–Esseen theorem),
    uses the Poisson approximation instead.
    """
    if special.comb(n, n / 2) != np.inf:
        if verbose:
            print('Exact solution found.')
        return stats.binom.cdf(x, n, p)

    # check for 3-sigma criterion
    if (n > 9 * (1 - p) / p) and (n > 9 * p / (1 - p)):
        # continuity correction from the addition of 1/2
        if verbose:
            print('Binomial CDF by normal approximation.')
        return stats.norm.cdf(x + .5, loc=n * p, scale=np.sqrt(n * p * (1 - p)))

    # check for np <= 1 criterion
    if verbose:
        print('Binomial CDF by Poisson approximation.')
    return stats.poisson.cdf(x, mu=n * p)

    if n * p > 1:
        warnings.warn('Binomial CDF could not be accurately estimated... Defaulting to Poisson approx.', RuntimeWarning)


def dijkstra(g, start, end):
    """
    Implementation of Dijkstra's algo for 2D grid as opposed to
    graphs as in scipy's implementation.

    Params
    ------
    g: ndarray
        weight matrix
    start: two elements array
        coordinates to starting node
    end: two elements array
        coordinates to end node

    Returns
    -------
    path: tuple of arrays
        array of x-y indices of shortest connecting path
    score: float
        score of shortest path

    e.g.
        x, y =  dijkstra(np.max(g) - g, [12, 6], [24, 8])
    """
    start = np.ravel_multi_index((start[0], start[1]), g.shape)
    end = np.ravel_multi_index((end[0], end[1]), g.shape)

    neighbourhood = np.array([
        [-1, -1, -1,  0, 0,  1, 1, 1],
        [-1,  0,  1, -1, 1, -1, 0, 1]
    ]).T

    flat_g = g.flatten()
    backtrace = np.zeros_like(flat_g, dtype=int)
    visited = np.zeros_like(flat_g, dtype=bool)
    score = np.ones_like(flat_g) * np.inf
    score[start] = 0

    current = np.argmin(score)
    while current != end:
        visited[current] = True

        neigh = np.squeeze(np.array(np.unravel_index([current], g.shape)))
        neigh = neighbourhood + neigh
        neigh = neigh[np.all(neigh >= 0, axis=1), :]
        neigh = neigh[np.all(neigh < g.shape, axis=1), :]
        neigh = np.ravel_multi_index((neigh[:, 0], neigh[:, 1]), g.shape)
        neigh = neigh[~visited[neigh]]
        if len(neigh) == 0:
            score[visited] = np.inf
            current = np.argmin(score)
            continue

        candidate = flat_g[neigh] + score[current]
        idx = np.argmin(np.array([score[neigh], candidate]), axis=0).astype(bool)
        score[neigh[idx]] = candidate[idx]
        backtrace[neigh[idx]] = current

        score[visited] = np.inf
        current = np.argmin(score)

    path = [current]
    while path[-1] != start:
        path.append(backtrace[path[-1]])

    return np.unravel_index(path, g.shape), score[current]


def squareform(X):
    """
    Get the upper triangle of any matrix
    """
    idx = np.ones(X.shape, dtype=bool)
    idx = np.triu(idx, 1)
    return X[idx]


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
    
