from . import utils
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def hmaps(behaviour: dict, spks: np.ndarray, bins=80, sigma=2) -> dict:
    """
    Place cells analysis routine.
    Create heat maps (i.e., rasters and stacks), compute SI
    """

    pos = behaviour['position']
    cum_trial = np.zeros_like(pos)
    cum_trial[behaviour['trial']] = 1
    cum_trial = np.cumsum(cum_trial)
    cum_trial[cum_trial >= (np.max(cum_trial) - 1)] = 0

    mvt = behaviour['movement']
    pos = pos[mvt]
    cum_trial = cum_trial[mvt]
    spks = spks[:, mvt]
    
    idx = np.array([pos, cum_trial]).T
    ranges = ([0, np.max(pos)], [0, np.max(cum_trial)])
    occ, _ = np.histogramdd(idx, range=ranges, bins=(bins, np.max(cum_trial).astype(int) + 1))
    rasters = np.array([np.histogramdd(idx, range=ranges, bins=(bins, np.max(cum_trial).astype(int) + 1), weights=spks[i, :])[0] for i in range(spks.shape[0])])
    rasters = rasters / occ
    rasters = rasters[:, :, 1:]
    rasters[np.isnan(rasters)] = 0
    srasters = utils.fast_smooth(rasters, 2, axis=1)
    
    idx = pos
    ranges = ([0, np.max(pos)],)
    occ, _ = np.histogramdd(idx, range=ranges, bins=bins)
    stack = np.array([np.histogramdd(idx, range=ranges, bins=bins, weights=spks[i, :])[0] for i in range(spks.shape[0])])
    stack = stack / occ
    stack[np.isnan(stack)] = 0
    sstack = utils.fast_smooth(stack, 2, axis=1)

    SI = calc_si(spks, pos)

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

def rasterize(spks, pos, trials):
    """
    Generate positional firing rate over trials rasters.
    Trials must be cumulative, with 0 being rejected. If
    trials is None, generate stacks instead.
    """

def shuffle_spks(spks, shift=None, mode='circ'):
    """
    Shuffle spike trains. Either 'circ' mode or 'burst' shuffler.
    If shift is None, shuffle by random factor.
    """

def calc_si(spks: np.ndarray, pos: np.ndarray, sigma=30, bins=80, KSG=True):
    """
    Calculate spatial information (i.e., mutual information)
    Between spike trains and position using the more robust
    KSG estimator. Alternatively, set KSG to False to use
    the traditional Skaggs method.
    """
    bins = np.linspace(0, np.max(pos), bins+1)
    pos = np.digitize(pos, bins=bins)
    spks = utils.fast_smooth(spks, sigma, axis=1)
    SI = mutual_info_classif(spks.T, pos, discrete_features=False, n_neighbors=3, n_jobs=-1)
    SI = SI * np.log2(np.exp(1)) # convert nats to bits
    return SI