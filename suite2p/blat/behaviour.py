import numpy as np
import h5py
import copy
from scipy import stats, signal, interpolate
from scipy.cluster.vq import kmeans
from scipy.optimize import dual_annealing
from sklearn.cluster import KMeans
from .utils import fast_smooth, knnsearch, fill_gaps, gethead, corr


_template = {
    'licks': np.array([], dtype=np.int64),
    'reward': np.array([], dtype=np.int64),
    'trial': np.array([], dtype=np.int64),
    'position': np.array([], dtype=np.float64),
    'velocity': np.array([], dtype=np.float64),
    'movement': np.array([], dtype=bool),
    'ts': np.array([], dtype=np.float64),
    'fs': None,
    'epochs': np.array([], dtype=np.int64), # 0 for rejection, 1 for rest, 2 for run, 3 for intertrial
}

def behaviour_template():
    return copy.deepcopy(_template)

def extract_behaviour(fn: str, v_range: list[float] = [-10.0, 10.0],
                      normalize: float = 180.0, bit_res: int = 12) -> dict:
    """
    Extract behaviour from ScanImage HDF5 file.

    Params
    ------
    fn          full path to .h5 file
    v_range     voltage range for position
    normalize   convert voltages to cm for position
    bit_res     bit resolution of position output

    Return
    ------
    behaviour   dictionary with fields 'position',
                'licks', 'reward', 'trial', 'ts'
                and 'fs'
    """

    with h5py.File(fn, 'r') as file:
        frame = file['Frame'][()]
        licks = file['Lick'][()]
        pos = file['Position'][()]
        reward = file['Reward'][()]
        trial = file['Trial'][()]
        fs = file.attrs['samplerate']

    frame_idx, _ = ts_extractor(frame, gapless=True)
    twop_fs = 1 / np.median(np.diff(frame_idx)) * fs # sampling rate of scope

    # filter out line noise
    discrete = np.linspace(v_range[0], v_range[1], num=2**bit_res)
    idx = knnsearch(discrete, pos)
    pos = discrete[idx]

    # compute cumulative position, i.e. correct for voltage mod(10)
    heads, tails = ts_extractor(pos)
    for h, t in zip(heads, tails):
        pos[t:] += pos[h-1] - pos[t]
        pos[h:t] = pos[t]
    heads, tails = ts_extractor(-pos)
    for h, t in zip(heads, tails):
        pos[t:] += pos[h-1] - pos[t]
        pos[h:t] = pos[t]

    # voltage to cm
    if normalize is not None:
        pos = (pos - v_range[0]) / np.diff(v_range) * normalize

    # smooth velocity
    vel = fast_smooth(pos, int(fs/twop_fs))
    vel = np.diff(vel) * fs # velocity

    #detect movements
    mvt = detect_mvt(vel, gaps=.25, fs=fs, prioritize='movement')

    # convert cumulative to raw positions from trials
    trial_idx, _ = ts_extractor(trial, si=5e3, thres=.5, polarity='positive', uniform=True)
    epochs = np.zeros_like(pos)
    reward_idx, _ = ts_extractor(reward, thres=2, polarity='positive')
    if reward_idx.shape[0] < 2:
        reward_idx = np.array([])
    if (trial_idx.shape[0] == 2) & (reward_idx.shape[0] == 0):
        print("automatically detected rest session")
        epochs[trial_idx[0]:trial_idx[1]] = 1
        trial_idx = np.array([])
    elif trial_idx.shape[0] > 2:
        print("automatically detected run session")
        # epochs[trial_idx[0]:trial_idx[-1]] = 2
        # trial_idx = trial_idx[:-1]
        for t in trial_idx:
            pos[t:] -= pos[t]

        trial_ids = np.round(trial[trial_idx + 10]).astype(int)
        trial_seq = np.unique(trial_ids)
        trial_seq = np.setdiff1d(trial_seq, [4])
        ptr = np.flatnonzero(trial_seq == 2)[0]
        idx = []
        for i in np.flatnonzero(trial_ids == trial_seq[0]):
            if len(trial_ids) - i < len(trial_seq) + 1:
                break
            if not np.all(trial_ids[i:i + len(trial_seq)] == trial_seq):
                raise RuntimeError("misordered trial sequence found, check raw trace for errors")
            epochs[ trial_idx[i]:trial_idx[i + ptr] ] = 3
            epochs[ trial_idx[i + ptr]:trial_idx[i + ptr + 1] ] = 2
            epochs[ trial_idx[i + ptr + 1]:trial_idx[i + len(trial_seq)] ] = 3
            idx.append(i)
        trial_idx = trial_idx[idx]

        reward_idx = knnsearch(frame_idx, reward_idx)
    else:
        raise RuntimeError("Corrupted trial indices. Automatic rest/run detection failed.")

    # licks_idx = ts_extractor(licks)
    trial_idx = knnsearch(frame_idx, trial_idx)
    # licks_idx = knnsearch(frame_idx, licks_idx)

    # decimate
    behaviour = behaviour_template()
    behaviour.update({
        # 'licks': licks_idx,
        'reward': reward_idx,
        'trial': trial_idx,
        'position': pos[frame_idx],
        'velocity': vel[frame_idx],
        'movement': mvt[frame_idx],
        'ts': frame_idx / fs,
        'fs': twop_fs,
        'epochs': epochs[frame_idx], # 0 for rejection, 1 for rest, 2 for run, 3 for intertrial
    })
    return behaviour


def stitch(behaviour: list[dict]) -> dict:
    """
    Stitch together a list of behaviours.
    """
    cum = ['reward', 'trial']
    cat = ['position', 'velocity', 'movement', 'epochs']

    homie = behaviour[0]
    for beh in behaviour[1:]:
        for key, field in homie.items():
            if key in cat:
                homie[key] = np.concatenate((homie[key], beh[key]))
            elif key in cum:
                homie[key] = np.concatenate((homie[key], beh[key] + homie['ts'].shape[0]))
            elif key == 'ts':
                homie[key] = np.concatenate((homie[key], beh[key] - beh[key][0] + 1/beh['fs'] + homie[key][-1]))

    return homie


def infer_motion(ops: list[dict], behaviours: list[dict], missing: list[bool] | np.ndarray,
                 block_lengths: list[int] | np.ndarray | None = None, planes: list[int] = [0],
                 nplanes: int = 1, override: bool = False) -> list[dict]:
    """
    If one of the behavioural recordings is corrupted or missing, infer movement
    from the imaging drift from registration. Can also be useful as a secondary
    source of movement detection.
    NOTE: this only works for rest sessions

    ===== Inputs =====
    ops: list of dict
        list of ops dicts from suite2p, ONE PER PLANE

    behaviours: list of dict
        list of extracted behaviour dicts

    missing: list of bool
        boolean index of missing behaviour sessions

    block_lengths: list of int
        number of frames for each block of recording (for a single plane)

    planes: list of int
        list of planes being analysed

    nplanes: int
        total number of planes

    override: bool
        whether to merge the registration-detected movements with the behaviour
        movememnts

    ===== Returns =====
    behaviours: list of dict
        complete list of behaviours including the missing ones

    """
    print('--- inferring animal motion from registration drift ---')
    behaviours = copy.deepcopy(behaviours)

    assert len(ops) == len(planes), 'ops must match planes'
    assert len(planes) <= nplanes, 'chosen more planes than total number of planes'
    missing = np.array(missing)
    assert np.sum(~missing) == len(behaviours), 'non-matching missing index and behaviours'

    ops = [o[()] for o in ops]
    ops = [np.sqrt(np.diff(o['xoff'])**2 + np.diff(o['yoff'])**2) for o in ops]
    for i in range(len(ops)):
        ops[i] = np.insert(ops[i], 0, [ops[i][0]])
    temp = np.empty((ops[0].shape[0] * len(ops),))
    for i, d in enumerate(ops):
        temp[i::len(ops)] = d
    drift = temp

    vel = np.concatenate([b['velocity'] for b in behaviours], axis=0)
    vel = [vel[p::nplanes] for p in planes]
    temp = np.empty((vel[0].shape[0] * len(vel),))
    for i, v in enumerate(vel):
        temp[i::len(vel)] = v
    vel = np.abs(temp)

    if block_lengths is None:
        block_lengths = [int(b['velocity'].shape[0] / nplanes) for b in behaviours]
        if np.sum(missing) == 1:
            block_lengths.insert(np.flatnonzero(missing)[0], int((drift.shape[0] - vel.shape[0]) / len(planes)))
        if np.sum(missing) > 1:
            lengths = np.unique(block_lengths)
            assert lengths.shape[0] == 1, 'block_lengths must be explicitly specified for non-uniform imaging blocks'
            for m in np.flatnonzero(missing):
                block_lengths.insert(m, int(lengths[0]))
    block_lengths = np.array(block_lengths)

    session = np.repeat(np.arange(len(missing)), block_lengths * len(planes))
    drift_blocks = [drift[session == i] for i in range(len(missing))]
    idx = np.isin(session, np.flatnonzero(~missing))
    drift = drift[idx]

    assert drift.shape[0] == vel.shape[0]

    rms = lambda x, wdw: np.sqrt(np.convolve(x**2, np.ones(wdw) / wdw, mode='same'))
    loss = lambda x, drift, vel: np.sum((drift > x[0]) ^ vel.astype(bool))

    tol = 5e-4
    wdw = 2
    r = -np.inf
    while True:
        wdw += 1
        pre = r
        r_drift = rms(drift, wdw)
        r = corr(r_drift, vel)[0, 0]**2
        if (r - pre) < tol:
            break

    ground_truth, _ = kmeans(vel, 2)
    ground_truth = np.argmin((vel[:, np.newaxis] - ground_truth[np.newaxis, :])**2, axis=1).astype(bool)
    if np.mean(vel[ground_truth]) < np.mean(vel[~ground_truth]):
        ground_truth = ~ground_truth
    ground_truth = fill_gaps(ground_truth, gap=wdw)

    thres = dual_annealing(loss, ((np.min(r_drift), np.max(r_drift)),), args=(r_drift, ground_truth))
    thres = thres['x'][0]

    inferred = [fill_gaps(rms(d, wdw) > thres, gap=wdw) for d in drift_blocks]
    for i, b in enumerate(block_lengths):
        fp = inferred[i]
        xp = np.arange(fp.shape[0])
        x = np.linspace(0, fp.shape[0], b * nplanes)
        f = interpolate.interp1d(xp, fp, kind='nearest-up', fill_value='extrapolate')
        inferred[i] = f(x)

    med_vel = np.median(vel[ground_truth])
    fs = np.median([b['fs'] for b in behaviours])
    for m in np.flatnonzero(missing):
        beh = behaviour_template()
        beh.update({
            'movement': inferred[m].astype(bool),
            'velocity': med_vel * inferred[m],
            'position': np.cumsum(med_vel * inferred[m]),
            'epochs': np.ones_like(inferred[m], dtype=int),
            'fs': fs,
            'ts': np.linspace(0, (inferred[m].shape[0] - 1) * fs, inferred[m].shape[0]),
        })
        behaviours.insert(m, beh)

    if override:
        for m in np.flatnonzero(~missing):
            behaviours[m]['movement'] |= inferred[m].astype(bool)

    return behaviours


def extract_plane(behaviour: dict, plane=0, nplanes=4) -> dict:
    """
    Extract behaviour from single plane.
    """
    behaviour = behaviour.copy()
    subsample = ['ts', 'position', 'velocity', 'movement', 'epochs']
    nearest = ['reward', 'trial']
    for key, field in behaviour.items():
        if key in subsample:
            behaviour[key] = behaviour[key][plane::nplanes]
        elif key in nearest:
            behaviour[key] = np.floor(behaviour[key] / nplanes).astype(int)
    behaviour['fs'] = 1 / np.median(np.diff(behaviour['ts']))

    return behaviour


def detect_mvt(vel, gaps=.25, fs=1.0, prioritize='movement'):
    """
    Detect movement epochs.

    Params
    ------
    gaps        fill in jitters to get continuous moving epochs
                if fs is not specified, this is the number of frames
    fs          sampling rate (optional)
    prioritize  either prioritize 'movement' or 'rest' to minimize the
                number of false negatives

    Return
    ------
    mvt         logical array of movement epochs
    """
    x = np.abs(vel)
    x = np.log(x[x > (np.max(vel)*1e-3)])
    kmeans = KMeans(n_clusters=2).fit(x.reshape(-1, 1))
    thres = np.abs(np.diff(kmeans.cluster_centers_.flatten())) / 2 + np.min(kmeans.cluster_centers_.flatten())
    thres = np.exp(thres)

    mvt = np.abs(vel) > thres
    if prioritize == "movement":
        mvt = fill_gaps(mvt, fs*gaps)
        mvt = ~fill_gaps(~mvt, fs*gaps)
    elif prioritize == "rest":
        mvt = ~fill_gaps(~mvt, fs*gaps)
        mvt = fill_gaps(mvt, fs*gaps)
    else:
        raise ValueError(prioritize + ' is not a valid option')

    return mvt


def ts_extractor(t, thres=.1, gapless=False, si=10, polarity=None, uniform=False):
    """
    Extract logical pulse indices exceeding the threshold.
    For super fast digitizers, analogue rise time can be slower
    than one sampling interval. This method corrects for those
    scenarios.
    Can use gapless if pulse intervals are uniform (e.g., frame
    pulses).
    Increase si, the gap factor, if digitizer is super duper
    fast.
    If pulses are expected to be of uniform length, set uniform
    True so that longer pulses due to noise are rejected.
    """
    if polarity == 'positive':
        t[t < -thres] = np.nan
    elif polarity == 'negative':
        t[t > thres] = np.nan
        t = -t

    if uniform:
        if polarity is None:
            warnings.warn('Polarity not defined with uniform pulse width. Assuming positive polarity.')
        heads, tails = gethead(t > thres), gethead(t > thres, tail=True)
        heads, tails = np.flatnonzero(heads), np.flatnonzero(tails)
        dur = tails - heads
        m, _ = stats.mode(dur)
        bad = (dur > m * 10) | (dur < m * .1)
        heads, tails = heads[bad], tails[bad]
        for hd, tl in zip(heads, tails):
            t[hd:tl + 1] = np.nan

    t = np.nan_to_num(t, copy=False, nan=0.0)

    idx = np.flatnonzero(np.diff(t) > thres)
    if gapless:
        si, _ = stats.mode(np.diff(idx))
        si = si / 2
    if idx.shape[0] == 1:
        good = np.array([-1])
    else:
        good = np.flatnonzero(np.diff(idx) > si)
    if good.shape[0] == 0:
        return np.array([]), np.array([])

    heads = idx[np.concatenate(([0], good + 1))]
    tails = idx[np.concatenate((good, [-1]))] + 1
    heads, tails = np.unique(heads), np.unique(tails)

    return heads, tails
