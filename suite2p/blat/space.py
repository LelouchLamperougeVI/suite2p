"""
Spatial analyses.
TODO:
- Implement SI Skaggs
- Trials stability place cells detection
- Burst shuffler
"""

from . import utils
from .KSG import ksg_mi
import numpy as np
from rich.progress import track

def pc_analysis(behaviour: dict, spks: np.ndarray, bins=80, sigma=2, nboots=1_000, alpha=.05) -> dict:
    """
    Place cells analysis routine.
    Create heat maps (i.e., rasters and stacks), compute SI
    """

    pos = behaviour['position']
    cum_trial = np.zeros_like(pos)
    cum_trial[behaviour['trial']] = 1
    cum_trial = np.cumsum(cum_trial)
    cum_trial[cum_trial == np.max(cum_trial)] = 0 # reject last trial

    mvt = behaviour['movement'] & (cum_trial != 0)
    pos = pos[mvt]
    cum_trial = cum_trial[mvt]
    spks = spks[:, mvt]

    silent = np.sum(spks, axis=1) == 0
    SI = np.zeros((spks.shape[0],))
    SI[~silent] = calc_si(spks[~silent, :], pos)
    if nboots is None:
        p_SI = np.array([None] * spks.shape[0])
    else:
        p_SI = np.zeros((spks.shape[0],))
        p_SI[~silent] = permutation_test(spks[~silent, :], func=calc_si, args=(pos,), nperms=500)

    rasters = rasterize(spks, pos=pos, trials=cum_trial, bins=bins)
    rasters[np.isinf(rasters)] = np.nan
    srasters = utils.fast_smooth(rasters, 2, axis=1)
    
    stack = rasterize(spks, pos, bins=bins)
    stack[np.isinf(stack)] = np.nan
    sstack = utils.fast_smooth(stack, 2, axis=1)

    stability = splithalf(srasters)
    if nboots is None:
        p_split = np.array([None] * spks.shape[0])
    else:
        boot = np.zeros((nboots, srasters.shape[0]))
        for i in range(nboots):
            sample = np.random.choice(srasters.shape[2], srasters.shape[0], replace=True)
            boot[i, :] = splithalf(srasters[:, :, sample])
        p_split = np.sum(boot < 0, axis=0) / nboots
        p_split[p_split == 0] = 1 / nboots

    ispc = (p_SI < alpha) & (p_split < alpha)

    ret = {
        'raw': {
            'rasters': rasters,
            'stack': stack,
        },
        'smooth': {
            'rasters': srasters,
            'stack': sstack,
        },
        'SI': SI,
        'stability': stability,
        'silent': silent,
        'ispc': ispc,
        'tests': {
            'SI': p_SI,
            'split': p_split,
        },
    }

    return ret


def rasterize(spks, pos, trials=None, bins=80):
    """
    Generate positional firing rate over trials rasters.
    Trials must be cumulative. If trials is None, generate
    stacks instead.
    """
    if trials is not None:
        idx = np.array([pos, trials]).T
        ranges = ([0, np.max(pos)], [np.min(trials), np.max(trials) + 1])
        bins=(bins, np.ptp(trials).astype(int) + 1)
    else:
        idx = pos
        ranges = ([0, np.max(pos)],)
    occ, _ = np.histogramdd(idx, range=ranges, bins=bins)
    rasters = np.array([np.histogramdd(idx, range=ranges, bins=bins, weights=spks[i, :])[0] for i in range(spks.shape[0])])
    rasters = rasters / occ
    rasters[np.isnan(rasters)] = 0

    return rasters


def splithalf(raster):
    x = np.mean(raster[:, :, ::2], axis=2).T
    y = np.mean(raster[:, :, 1::2], axis=2).T
    r = np.sum( (x - np.mean(x, axis=0)) * (y - np.mean(y, axis=0)), axis=0 ) /\
        np.sqrt( np.sum((x - np.mean(x, axis=0))**2, axis=0) * np.sum((y - np.mean(y, axis=0))**2, axis=0) )
    return r


def permutation_test(spks: np.ndarray, func: callable, nperms=500, mode='burst', args=tuple()):
    """
    Shuffle spike trains. Either 'circ' mode or 'burst' shuffler.
    If shift is None, shuffle by random factor. Calls func at each
    permutation. func must be of form fun(spks, *args) and must
    return an array of shape (spks.shape[0],).
    """
    truth = func(spks, *args)
    perms = np.zeros((nperms, truth.shape[0]))
    for i in track(range(nperms), description='permutation testing...'):
        if mode == 'circ':
            shift = np.random.randint(spks.shape[1])
            spks = np.roll(spks, shift, axis=1)
        elif mode == 'burst':
            spks = burst_shuffler(spks)
        perms[i, :] = func(spks, *args)

    p = np.sum(truth < perms, axis=0) / nperms
    p[p == 0] = 1 / nperms
    
    return p


def burst_shuffler(spks: np.ndarray):
    spks = spks.copy()
    for i, train in enumerate(spks):
        isi = np.diff(np.flatnonzero(train))
        np.random.shuffle(isi)
        np.random.shuffle(train)
        isi = np.cumsum(np.concatenate(([0], isi)))
        shuffled = np.zeros_like(train)
        shuffled[isi] = train[train != 0]
        shuffled = np.roll(shuffled, np.random.randint(train.shape[0]))
        spks[i, :] = shuffled

    return spks


def calc_si(spks: np.ndarray, pos: np.ndarray, sigma=30, KSG=True):
    """
    Calculate spatial information (i.e., mutual information)
    Between spike trains and position using the more robust
    KSG estimator. Alternatively, set KSG to False to use
    the traditional Skaggs method.
    """
    spks = utils.fast_smooth(spks, sigma, axis=1)
    SI = ksg_mi(spks, pos, k=5)
    return SI
