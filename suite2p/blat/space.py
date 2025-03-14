"""
Spatial analyses.
TODO:
- Implement SI Skaggs
- Trials stability place cells detection
- Burst shuffler
"""

from . import utils
# from .KSG import ksg_mi
from libKSG import KSG
from multiprocessing import Pool
import threading
import numpy as np
from rich.progress import Progress

def pc_analysis(behaviour: dict, spks: np.ndarray, bins=80, sigma=2, nboots=1_000, alpha=.05, ksg_sigma=30) -> dict:
    """
    Place cells analysis routine.
    Create heat maps (i.e., rasters and stacks), compute SI
    """

    pos = behaviour['position']
    cum_trial = np.zeros_like(pos)
    cum_trial[behaviour['trial']] = 1
    cum_trial = np.cumsum(cum_trial)
    cum_trial[cum_trial == np.max(cum_trial)] = 0 # reject last trial

    mvt = behaviour['movement'] & (cum_trial != 0) & (behaviour['epochs'] == 2)
    pos = pos[mvt]
    cum_trial = cum_trial[mvt]
    cum_trial = np.cumsum(np.diff(cum_trial) > 0)
    cum_trial = np.insert(cum_trial, 0, 0)
    spks = spks[:, mvt]

    silent = np.sum(spks, axis=1) == 0
    SI = np.zeros((spks.shape[0],))
    SI[~silent] = calc_si(spks[~silent, :], pos, sigma=ksg_sigma)
    if nboots is None:
        p_SI = np.array([1] * spks.shape[0])
    else:
        p_SI = np.ones((spks.shape[0],))
        p_SI[~silent] = permutation_test(spks[~silent, :], pos, ksg_sigma=ksg_sigma, nperms=500)

    rasters = rasterize(spks, pos=pos, trials=cum_trial, bins=bins)
    rasters[np.isinf(rasters)] = np.nan
    srasters = utils.fast_smooth(rasters, 2, axis=1)

    stack = rasterize(spks, pos, bins=bins)
    stack[np.isinf(stack)] = np.nan
    sstack = utils.fast_smooth(stack, 2, axis=1)

    stability = splithalf(srasters)
    if nboots is None:
        p_split = np.array([1] * spks.shape[0])
    else:
        boot = np.zeros((nboots, srasters.shape[0]))
        for i in range(nboots):
            sample = np.random.choice(srasters.shape[2], srasters.shape[0], replace=True)
            boot[i, :] = splithalf(srasters[:, :, sample])
        p_split = np.sum(boot < 0, axis=0) / nboots
        p_split[p_split == 0] = 1 / nboots
    p_split[silent] = 1

    ispc = (p_SI <= alpha) & (p_split <= alpha)

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


def _si_initializer():
    global ksg_obj
    ksg_obj = KSG()

def _si_job_circ(x):
    (spks, pos, sigma) = x
    shift = np.random.randint(spks.shape[1])
    spks = np.roll(spks, shift, axis=1)
    spks = utils.fast_smooth(spks, sigma=sigma, axis=1)
    return ksg_obj.mi(spks, pos)

def _si_job_burst(x):
    (spks, pos, sigma) = x
    spks = burst_shuffler(spks)
    spks = utils.fast_smooth(spks, sigma=sigma, axis=1)
    return ksg_obj.mi(spks, pos)

def permutation_test(spks: np.ndarray, pos: np.ndarray, ksg_sigma=30, nperms=500, mode='burst'):
    """
    Shuffle spike trains. Either 'circ' mode or 'burst' shuffler.
    If shift is None, shuffle by random factor. Calls func at each
    permutation. func must be of form fun(spks, *args) and must
    return an array of shape (spks.shape[0],).
    UPDATE: only calculates SI now.....
    """
    truth = calc_si(spks, pos, sigma=ksg_sigma)
    # perms = np.zeros((nperms, truth.shape[0]))
    # for i in track(range(nperms), description='permutation testing...'):
    pool = Pool(initializer=_si_initializer)
    if mode == 'circ':
        workload = pool.imap_unordered(_si_job_circ, [(spks, pos, ksg_sigma) for _ in range(nperms)])
    elif mode == 'burst':
        workload = pool.imap_unordered(_si_job_burst, [(spks, pos, ksg_sigma) for _ in range(nperms)])

    perms = []
    with Progress() as progress:
        task_id = progress.add_task('permutation testing...', total=nperms)
        for ret in workload:
                perms.append(ret)
                progress.advance(task_id)
    perms = np.array(perms)

    p = 1 - np.sum(truth > perms, axis=0) / nperms
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


def calc_si(spks: np.ndarray, pos: np.ndarray, sigma=30):
    """
    Calculate spatial information (i.e., mutual information)
    Between spike trains and position using the more robust
    KSG estimator. Alternatively, set KSG to False to use
    the traditional Skaggs method.
    """
    spks = utils.fast_smooth(spks, sigma, axis=1)
    ksg = KSG()
    SI = ksg.mi(spks, pos, k=5)
    SI[SI < 0] = 0
    return SI
