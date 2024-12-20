import numpy as np
import h5py
from scipy import stats, signal
from sklearn.cluster import KMeans
from .utils import fast_smooth, knnsearch, fill_gaps, gethead

def extract_behaviour(fn, v_range=[-10.0, 10.0], normalize=180.0, bit_res=12) -> dict:
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
    behaviour = {
        # 'licks': licks_idx,
        'reward': reward_idx,
        'trial': trial_idx,
        'position': pos[frame_idx],
        'velocity': vel[frame_idx],
        'movement': mvt[frame_idx],
        'ts': frame_idx / fs,
        'fs': twop_fs,
        'epochs': epochs[frame_idx], # 0 for rejection, 1 for rest, 2 for run, 3 for intertrial
    }
    return behaviour


def stitch(behaviour: list) -> dict:
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
