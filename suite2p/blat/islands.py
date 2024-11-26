import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.optimize import minimize_scalar
from suite2p.blat import utils
from rich import progress

def r2bin(r, mode='ev'):
    """
    Convert correlation matrix to binary matrix.
    Threshold set by preserving the maximum explained
    variance in the ogirinal matrix (mode='ev').
    Alternatively, set the threshold by preserving
    the maximum entropy (mode='entropy').
    """
    coefs = r[np.triu(np.ones_like(r).astype(bool), 1)]
    if mode == 'ev':
        cost = lambda t: 1 - utils.corr(coefs, (coefs > t).astype(np.float64))[0][0]**2
    elif mode == 'entropy':
        cost = lambda t: np.nanmin([1 + np.mean(coefs > t) * np.log2(np.mean(coefs > t)) + np.mean(coefs <= t) * np.log2(np.mean(coefs <= t)), 1])
    else:
        raise ValueError('mode ' + mode + ' is undefined.')
    ops = minimize_scalar(fun=cost, bounds=[-1, 1])
    thres = ops['x']
    return r > thres


def mi_mat(x, norm=True, axis=1):
    """
    Compute the mutual information matrix.
    If norm, normalize the matrix by max(H(a), H(b)).
    """
    if axis == 0:
        x = x.T
        
    I = np.zeros((x.shape[0], x.shape[0]))
    for i in progress.track(range(I.shape[0]), description='MI matrix with KSG estimator...'):
        I[i, i:] = ksg_mi(x[i:, :], x[i, :])
    I = I + np.rot90(np.flipud(np.triu(I, 1)), k=3)

    if norm:
        H = np.diag(I)
        H = np.array([[np.max([i, j]) for j in H] for i in H])
        I = I / H

    return I


def bmf(d, patterns = None, count = 0, thres = .99, maxIter=500):
    """
    Binary matrix factorization.
    https://arxiv.org/abs/1909.03991
    """
    if patterns is None:
        patterns = np.zeros((1, d.shape[0]))
    if np.sum(d) == 0 or count == maxIter:
        patterns = np.delete(patterns, -1, axis = 0)
        return d, patterns
    idx = np.argsort(np.sum(d, axis = 0))
    idx = idx[::-1]
    idx_ = np.argsort(idx)
    d_ = d[np.ix_(idx, idx)]
      
    m = np.ceil(np.max(np.argwhere(np.sum(d_, axis = 0) > 0)) / 2).astype(int)
    idx = np.sum(np.logical_and(d_[:, m], d_), axis = 1) / np.sum(d_[:, m]) > thres
    
    if np.sum(idx) > 1: # do not consider noise vectors
        pattern, _ = stats.mode(d_[:, idx].astype(int), axis = 1)
        patterns[-1, :] = pattern[idx_]
        patterns = np.vstack((patterns, np.zeros((1, d_.shape[0]))))
      
    d_[:, idx] = False
    d_[idx, :] = False
    d_ = d_[np.ix_(idx_, idx_)]
      
    return bmf(d_, patterns, count+1)


def binothres(d, prct = .99):
    """
    Estimate threshold by modelling as binomial dist.
    """
    p = np.mean(np.sum(d, axis = 0)) / d.shape[0]
    c = stats.binom.ppf(prct, d.shape[0], p**2)
    return c / np.floor(p * d.shape[0])


def prune(patterns, nmin=5, overlap=.75):
    """
    Prune the patterns, merge overlaps and
    delete small ensembles.
    """
    patterns = patterns.astype(bool).copy()
    patterns = patterns[np.sum(patterns, axis=1) >= nmin, :]
    
    jaccard = np.sum(patterns[:, :, np.newaxis] & patterns.T[np.newaxis, :, :], axis=1)
    jaccard = jaccard / \
                np.min(np.stack(\
                    (np.tile(np.sum(patterns, axis=1)[:, np.newaxis], reps=(1, patterns.shape[0])), \
                     np.tile(np.sum(patterns, axis=1)[np.newaxis, :], reps=(patterns.shape[0], 1))), \
                    axis=2), axis=2)
    jaccard = np.triu(jaccard, 1)

    x, y = np.unravel_index(np.argmax(jaccard), jaccard.shape)
    if jaccard[x, y] < overlap:
        return patterns
    patterns[x, :] = patterns[x, :] | patterns[y, :]
    patterns = np.delete(patterns, obj=y, axis=0)

    patterns = prune(patterns, nmin=nmin, overlap=overlap)
    return patterns
        

def sort(patterns):
    patterns = patterns.astype(bool).copy()
    jaccard = np.sum(patterns[:, :, np.newaxis] & patterns.T[np.newaxis, :, :], axis=1)
    jaccard = jaccard / (np.sum(patterns, axis=1)[:, np.newaxis] + np.sum(patterns, axis=1)[np.newaxis, :] - jaccard)
    jaccard[np.diagflat(np.diag(np.ones_like(jaccard).astype(bool)))] = 0
    
    lsorted = np.zeros((patterns.shape[0],)).astype(bool)
    current = np.argmax(np.sum(patterns, axis=1))
    idx = 0
    order = np.arange(patterns.shape[1])
    while ~np.all(lsorted):
        linked = np.argmax(jaccard[current, :])
        if jaccard[current, linked] == 0:
            seq = ~patterns[current, idx:]
        else:
            seq = (~patterns[current, idx:]).astype(int) + patterns[linked, idx:].astype(int) + \
                    (~(patterns[current, idx:] | patterns[linked, idx:])).astype(int) * 2
        sorting = np.argsort(seq)
        temp = order[idx:]
        order[idx:] = temp[sorting]
        temp = patterns[:, idx:]
        patterns[:, idx:] = temp[:, sorting]
        idx = np.flatnonzero(patterns[current, :])[-1] + 1
        lsorted[current] = True
        if jaccard[current, linked] == 0:
            jaccard[:, current] = 0
            current = np.argmax(np.sum(patterns, axis=1) * ~lsorted)
        else:
            jaccard[:, current] = 0
            current = linked

    # patterns = patterns[np.argsort(np.sum(patterns, axis=1))[::-1], :]
    
    return patterns, order
