"""
Spatial analyses.
TODO:
- SI place cells detection
- Trials stability place cells detection
"""

from . import utils
from .KSG import ksg_mi
import numpy as np
from tqdm import tqdm

def hmaps(behaviour: dict, spks: np.ndarray, bins=80, sigma=2) -> dict:
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

    rasters = rasterize(spks, pos=pos, trials=cum_trial, bins=bins)
    srasters = utils.fast_smooth(rasters, 2, axis=1)
    
    stack = rasterize(spks, pos, bins=bins)
    sstack = utils.fast_smooth(stack, 2, axis=1)

    silent = np.sum(spks, axis=1) == 0
    SI = np.zeros_like(silent)
    SI[~silent] = calc_si(spks[~silent, :], pos)
    p = np.ones_like(silent)
    p[~silent] = permutation_test(spks[~silent, :], func=calc_si, args=(pos,), nperms=500)
    print(p)

    ret = {
        'rasters': rasters,
        'stack': stack,
        'smooth': {
            'rasters': srasters,
            'stack': sstack,
        },
        'SI': SI,
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

def permutation_test(spks: np.ndarray, func: callable, nperms=1_000, mode='circ', args=tuple()):
    """
    Shuffle spike trains. Either 'circ' mode or 'burst' shuffler.
    If shift is None, shuffle by random factor. Calls func at each
    permutation. func must be of form fun(spks, *args) and must
    return an array of shape (spks.shape[0],).
    """
    truth = func(spks, *args)
    if mode == 'circ':
        perms = np.zeros((nperms, truth.shape[0]))
        for i in tqdm(range(nperms), desc='permutation testing...'):
            shift = np.random.randint(spks.shape[1])
            spks = np.roll(spks, shift, axis=1)
            perms[i, :] = func(spks, *args)

    p = np.sum(truth > perms, axis=0) / nperms
    p[p == 0] = 1/nperms
    
    return p

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
