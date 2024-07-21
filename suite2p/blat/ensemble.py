import numpy as np
from scipy import stats
from scipy.sparse import linalg
from suite2p.blat import utils, islands, space, core
import copy
import matplotlib.pyplot as plt

normalize = lambda x: (x.T / np.std(x, axis=1)).T

def mkmvt(plane, mode='run'):
    mvt = plane.behaviour['movement']
    epochs = plane.behaviour['epochs']
    if mode == 'run':
        mvt = mvt & (epochs == 2)
    elif mode == 'rest':
        if np.any(plane.behaviour['epochs'] == 1):
            mvt = ~mvt & (epochs == 1)
        else:
            mvt = ~mvt
    elif mode == 'valid':
        mvt = epochs > 0
    elif mode == 'all':
        mvt = np.ones_like(mvt).astype(bool)
    else:
        raise ValueError(mode + ' is undefined')

    return mvt
    

def plot(plane, patterns, order, mode='run', sigma=3):
    stack = plane.analysis['smooth']['stack']
    spks = plane.spks
    pos = plane.behaviour['position']
    mvt = plane.behaviour['movement']
    mvt = mkmvt(plane, mode=mode)

    spks = spks[:, mvt]
    _, R = extract(plane, patterns, mode=mode, sigma=sigma)
    
    spks = spks[order, :]
    patterns = patterns[:, order]
    for p in patterns:
        idx = np.argsort(np.argmax(stack[order[p], :], axis=1))
        temp = spks[p, :]
        spks[p, :] = temp[idx, :]
    
    p = ~np.any(patterns, axis=0)
    idx = np.argsort(np.argmax(stack[order[p], :], axis=1))
    temp = spks[p, :]
    spks[p, :] = temp[idx, :]

    spks = utils.fast_smooth(spks, axis=1, sigma=sigma)
    spks = normalize(spks)
    spks[np.isnan(spks)] = 0

    fig, axs = plt.subplots(3, 2, width_ratios=[1, 20], height_ratios=[3, 1, 1])
    axs[0, 0].imshow(patterns.T, aspect='auto', interpolation='none')
    axs[0, 1].imshow(spks, aspect='auto', interpolation='none')
    axs[0, 1].sharey(axs[0, 0])
    axs[1, 1].plot(R.T)
    axs[1, 1].sharex(axs[0, 1])
    axs[2, 1].plot(pos[mvt])
    axs[2, 1].sharex(axs[0, 1])
    

def pop2ens(analysis, patterns):
    """
    Turn population level pc_analysis into
    ensemble level pc_analysis.
    """

def extract(plane, patterns, mode='run', sigma=3):
    """
    Extract ensemble activation time courses and
    neuron weights from planepack object for the
    ensembles defined in patterns.
    Available modes are 'run', 'rest' and 'all'.
    """
    if type(plane) is not core.planepack:
        raise TypeError('Input is not a planepack object.')
        
    A = plane.spks
    mvt = plane.behaviour['movement']
    mvt = mkmvt(plane, mode=mode)
    
    A = A[:, mvt]
    A = utils.fast_smooth(A, axis=1, sigma=sigma)
    A = normalize(A)
    A[np.isnan(A)] = 0
    
    W = np.zeros((A.shape[0], patterns.shape[0]))
    R = np.zeros((patterns.shape[0], A.shape[1]))
    for i in range(patterns.shape[0]):
        u, s, v = linalg.svds(A[patterns[i, :], :], k=1)
        u = u[:, 0]
        v = v[0, :]
        if np.all(u <= 0):
            u = -u
            v = -v
        elif np.all(u >= 0):
            pass
        else:
            raise RuntimeError('Non-negative constraint not satisfied. Failed to solve constrained low-rank with SVD (Eckart–Young–Mirsky).')
        W[patterns[i, :], i] = u * s
        R[i, :] = v

    return W, R


def detect(plane, mode='trials', sigma=0, nmin=10, overlap=.5):
    """
    Detect ensembles in planepack object.

    Inputs
    ------
    plane: planepack object
    mode: choice between 'run', 'rest', 'all' and 'trials'
    
    """
    if type(plane) is not core.planepack:
        raise TypeError('Input is not a planepack object.')

    spks = normalize(plane.spks)
    mvt = plane.behaviour['movement']
    pos = plane.behaviour['position']
    trials = plane.behaviour['trial']
    
    cum_trials = np.zeros_like(pos)
    cum_trials[trials] = 1
    cum_trials = np.cumsum(cum_trials)

    if mode == 'trials':
        mu = np.array([utils.accumarray(cum_trials[mvt].astype(int), s[mvt], func=np.mean) for s in spks])
        mu = mu[:, 1:]
        mu = mu / np.sum(mu, axis=0)
    else:
        mvt = mkmvt(plane, mode=mode)
        mu = spks[:, mvt]
        mu = utils.fast_smooth(mu, sigma=sigma, axis=1)
        mu = normalize(mu)
    
    r = utils.corr(mu, axis=1)
    bm = islands.r2bin(r)
    _, patterns = islands.bmf(bm, thres = islands.binothres(bm))
    patterns = islands.prune(patterns, nmin=nmin, overlap=overlap)
    _, order = islands.sort(patterns)

    return patterns, order