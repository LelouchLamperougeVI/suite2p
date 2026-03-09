import numpy as np
from scipy import optimize, signal, cluster, stats, special
from statistics import NormalDist
from multiprocessing import Pool
from tqdm.auto import tqdm
import warnings
import copy
from suite2p.blat import utils

def _multup(P, raster, tol=1e-2, maxiter=10):
    """
    Find T by multiplicative update rules
    """
    T = np.ones((P.shape[1], raster.shape[1]))

    delta = [np.inf,]
    PR = P.T @ raster
    PP = P.T @ P
    for i in range(maxiter):
        W = PR / (PP @ T)
        W[np.isnan(W)] = 0
        W[np.isinf(W)] = 1
        T = T * W

        delta.append(np.linalg.norm(W, 2))
        if delta[-2] - delta[-1] < tol:
            break

    return T

def _fit_PT(x, raster):
    """
    Generate P and T from parameters x
    """
    assert len(x) % 2 == 0, 'number of params for optimization must be even'
    n_comps = len(x) // 2
    gauss = lambda x, b, sig: np.exp(-(x - b)**2 / (2 * sig**2))

    s = np.arange(raster.shape[0])
    P = np.empty((raster.shape[0], n_comps))
    for i in range(n_comps):
        P[:, i] = gauss(s, x[i*2], x[i*2 + 1])
    T = _multup(P, raster)

    return P, T

def _loss(x, raster):
    """
    Calculate reconstruction loss (L2) for global optimization.
    """
    P, T = _fit_PT(x, raster)
    return np.linalg.norm(raster - P @ T, 2)

def _decomp_unpack(args):
    return decompose_raster(*args)

def decompose_raster(raster, max_overlap=.2, min_gain=.1):
    """
    Decompose raster into place cells (P) and trials (T) components
    such that raster ~ P @ T.
    Assume that place cells are gaussians...
    """
    mu = np.mean(raster, axis=1)
    peaks, prop = signal.find_peaks(mu, height=0, width=0)
    init = [[a, b/2] for a, b, in zip(peaks, prop['widths'])]
    L = [_loss(x, raster) for x in init]
    order = np.argsort(L)
    init = np.array([init[i] for i in order]).flatten()
    if len(init) == 0:
        warnings.warn('empty raster, nothing to fit')
        P = np.zeros((raster.shape[0], 1))
        T = np.zeros((1, raster.shape[1]))
        return (P, T), np.nan

    ev = [0,]
    x = []
    for n_comps in range(1, len(init) // 2 + 1):
        bounds = [(0, raster.shape[0] - 1), (.1, raster.shape[0] / 4)] * n_comps
        res = optimize.dual_annealing(_loss, bounds, args=(raster,), x0=init[:n_comps*2], maxiter=300)
        ev.append(1 - res.fun / np.linalg.norm(raster))
        x.append(res.x)

        overlap = [[NormalDist(mu=m1, sigma=s1).overlap(NormalDist(mu=m2, sigma=s2)) \
                    for m1, s1 in zip(x[-1][::2], x[-1][1::2])] for m2, s2 in zip(x[-1][::2], x[-1][1::2])]
        overlap = np.triu(np.array(overlap), 1)
        if np.any(overlap > max_overlap):
            break
        if ev[-1] - ev[-2] < min_gain:
            break

    if len(x) == 1:
        warnings.warn('low explained variance, returning model after one iteration, are you sure this is a place cell?')
        P, T = _fit_PT(x[-1], raster)
        return (P, T), ev[-1]

    P, T = _fit_PT(x[-2], raster)
    return (P, T), ev[-2]

def decompose_rasters(rasters, max_overlap=.2, min_gain=.1, parallel=True):
    rasters = copy.deepcopy(rasters)
    rasters[rasters < 0] = 0

    if parallel:
        with Pool() as pool:
            ret = list(tqdm(pool.imap(_decomp_unpack, ((r, max_overlap, min_gain) for r in rasters)), \
                            total=rasters.shape[0], desc='non-negative marginalized Gaussian decomposition'))
    else:
        ret = [decompose_raster(r, max_overlap=max_overlap, min_gain=min_gain) \
               for r in tqdm(rasters, desc='non-negative marginalized Gaussian decomposition')]

    return [P for (P, T), ev in ret], [T for (P, T), ev in ret], [ev for (P, T), ev in ret]

def remap_epochs(t, alpha=.05):
    '''
    Find epochs when remapping occurred.
    '''
    cutoff = np.ptp(t) / 2 * (1 - alpha)

    splits = []
    candidates = set(range(1, len(t)))
    cost = [np.sum(np.abs(t - np.mean(t))),]
    while len(candidates) > 0:
        SAD = [np.sum([np.sum(np.abs(c - np.mean(c))) for c in np.split(t, np.sort(splits + [i,]))]) for i in candidates]
        idx = np.argmin(SAD)
        splits.append(idx)
        cost.append(np.min(SAD))
        candidates -= {idx}
        if (cost[-2] - cost[-1]) <= cutoff:
            splits.pop()
            cost.pop()
            break

    return splits

def remap_prob(t):
    '''
    Verify if remapping occurred using custom gap statistic.
    '''
    n = len(t)
    ba = np.ptp(t)

    ex_ba = lambda n, ba: ba / 2 -  ba / np.sqrt(n * 6 * np.pi)
    ex_n = lambda n: n * special.erf(np.sqrt(3 / 2 / (n - 1))) \
                        - 2 * n * np.sqrt(n - 1) / np.sqrt(6 * np.pi) * (1 - np.exp(-3 / 2 / (n - 1))) \
                        - 2 * n / np.sqrt(6 * np.pi * (n - 1)) + n / 2
    ex_w = lambda n, ba: (3 * n - 2) * ba / 12
    theo = lambda n, ba: ex_w(np.ceil(ex_n(n)), ex_ba(n, ba)) + ex_w(n - np.ceil(ex_n(n)), ba - ex_ba(n, ba))

    gap = [np.log(ex_w(n, ba)) - np.log(np.sum(np.abs(t - np.mean(t)))),]

    cent, _ = cluster.vq.kmeans(t, 2)
    idx = np.argmin(np.abs(cent[:, np.newaxis] - t[np.newaxis, :]), axis=0)
    SAD = np.sum([np.sum(np.abs(t[idx == i] - np.mean(t[idx == i]))) for i in range(2)])

    ex = np.log(theo(n, ba)) - np.log(SAD)
    gap.append(ex)

    return gap[0] < gap[1]

def find_remap(T, sigma=2):
    '''
    Find remapping events for individual place fields.
    '''
    remap = [[remap_prob(utils.fast_smooth(t, sigma=sigma)) for t in tt] for tt in T]
    epochs = [[remap_epochs(t) for t in tt] for tt in T]
    for i in range(len(remap)):
        for j in range(len(remap[i])):
            if len(epochs[i][j]) == 0:
                remap[i][j] = False
            if ~remap[i][j]:
                epochs[i][j] = []
    return remap, epochs


def detect_modules(epochs, window=3, alpha=.05):
    '''
    Detect remapping modules.
    '''
    # linearize and sort split epochs and associated neuron/place field ids
    linear = np.array([[i*10 + j, x] for i, xxx in enumerate(epochs) for j, xx in enumerate(xxx) for x in xx])
    neur = linear[:, 0]
    linear = linear[:, 1]
    idx = np.argsort(linear)
    linear = linear[idx]
    neur = neur[idx]
    d = np.abs(linear[:, np.newaxis] - linear[np.newaxis, :]) # distance matrix between remapping epochs

    # determine minimum number of members per module based on poisson distribution
    mu = len(linear) / np.max(linear) * window
    min_members = stats.poisson.isf(alpha, mu=mu).astype(int)

    # find squares or "boxes" in the distance matrix to classify as modules
    boxes = [np.flatnonzero(dd)[-1] for dd in d < 4] - np.arange(d.shape[0])
    boxes[boxes < min_members] = 0
    pks, _ = signal.find_peaks(boxes)

    # check if modules contain enough unique members
    modules = [set(neur[p:p+boxes[p]]) for p in pks]
    modules = [m for m in modules if len(m) >= min_members]

    # merge highly overlapping modules according to fisher exact test
    i = 0
    j = 1
    while (i < len(modules)) & (j < len(modules)):
        contingency = [[len(np.unique(neur)) - len(modules[i].union(modules[j])), len(modules[i].difference(modules[j]))],
                       [len(modules[j].difference(modules[i])), len(modules[i].intersection(modules[j]))]]
        pval = stats.fisher_exact(contingency, 'greater')
        if pval.pvalue < alpha:
            modules[i] = modules[i].union(modules[j])
            modules.remove(modules[j])
            i += j // len(modules)
            j = max(j % len(modules), i + 1)
            continue
        i += (j + 1) // len(modules)
        j = max((j + 1) % len(modules), i + 1)

    modules = [list(m) for m in modules]
    modules = [np.array([[c // 10, c % 10] for c in m]) for m in modules]
    return modules

