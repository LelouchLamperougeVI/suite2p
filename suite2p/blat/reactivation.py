from suite2p.blat.behaviour import detect_mvt
from suite2p.blat import utils
import numpy as np
import copy

def detect_rest(behaviour, gaps=10):
    """
    Define resting periods in behaviour dict,
    while getting rid of movement epochs more
    thoroughly.
    """
    
    epochs = behaviour['epochs']
    vel = behaviour['velocity']
    fs = behaviour['fs']
    
    rest_epochs = (np.cumsum(np.diff(epochs.astype(bool))) % 3) * epochs[1:]
    rest_epochs = np.insert(rest_epochs, 0, [0])

    for i in np.unique(rest_epochs)[1:]:
        mvt = detect_mvt(vel[rest_epochs == i], gaps=gaps, fs=fs, prioritize='movement')
        mvt = np.convolve(mvt, np.ones((int(np.round(gaps*fs)),)), mode='same').astype(bool)
        rest_epochs[rest_epochs == i] = ~mvt * i
        
    return rest_epochs

def ev(plane, rest_epochs, reference=2, control=None, sigma=2):
    """
    Calculate explained variance (i.e., partial
    correlation) between rest and run.
    
    Inputs:
    -------
    plane:        planepack object
    rest_epochs:  from detect_rest()
    reference:    the rest period to correlate against run
    control:      the rest period to serve as control,
                  all other rest periods if None
    sigma:        Gaussian smoothing factor in seconds

    Outputs:
    -------
    fev, rev:     forward and reverse explained variance
    """
    if control is None:
        control = (rest_epochs > 0) & (rest_epochs != reference)
    else:
        control = rest_epochs == control
        
    spks = copy.deepcopy(plane.spks)
    mvt = plane.behaviour['movement'] & (plane.behaviour['epochs'] == 1)
    fs = plane.behaviour['fs']
    
    X = utils.fast_smooth(spks[:, mvt], sigma=sigma*fs, axis=1)
    Y = utils.fast_smooth(spks[:, rest_epochs == reference], sigma=sigma*fs, axis=1)
    Z = utils.fast_smooth(spks[:, control], sigma=sigma*fs, axis=1)

    idx = np.ones((spks.shape[0], spks.shape[0]))
    idx = np.triu(idx, 1).astype(bool)

    X = utils.corr(X, axis=1)[idx]
    Y = utils.corr(Y, axis=1)[idx]
    Z = utils.corr(Z, axis=1)[idx]

    XY = utils.corr(X, Y)
    XZ = utils.corr(X, Z)
    ZY = utils.corr(Z, Y)

    fev = (XY - XZ * ZY) / (np.sqrt(1 - XZ**2) * np.sqrt(1 - ZY**2))
    rev = (XZ - XY * ZY) / (np.sqrt(1 - XY**2) * np.sqrt(1 - ZY**2))

    fev = fev.squeeze()
    rev = rev.squeeze()
    
    return fev, rev