import numpy as np
import h5py
from scipy import stats, signal
from .utils import fast_smooth, knnsearch

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
    kernel = np.ones((int(fs/twop_fs*2),)) / np.round(fs/twop_fs*2)
    vel = signal.fftconvolve(pos, kernel, mode='same') / signal.fftconvolve(np.ones_like(pos), kernel, mode='same')
    vel = np.diff(vel) * fs # velocity
    
    # convert cumulative to raw positions from trials
    trial_idx, _ = ts_extractor(trial)
    trial_idx = trial_idx[:-1]
    for t in trial_idx:
        pos[t:] -= pos[t]

    reward_idx, _ = ts_extractor(reward)
    # licks_idx = ts_extractor(licks)
    reward_idx = knnsearch(frame_idx, reward_idx)
    trial_idx = knnsearch(frame_idx, trial_idx)
    # licks_idx = knnsearch(frame_idx, licks_idx)

    vel = vel[frame_idx]
    # vel = signal.savgol_filter(vel, window_length=int(twop_fs*.1), polyorder=1)
    
    # decimate
    behaviour = {
        # 'licks': licks_idx,
        'reward': reward_idx,
        'trial': trial_idx,
        'position': pos[frame_idx],
        'velocity': vel,
        'ts': frame_idx / fs,
        'fs': twop_fs
    }
    return behaviour

def ts_extractor(t, thres=.1, gapless=False, si=10):
    """
    Extract logical pulse indices exceeding the threshold.
    For super fast digitizers, analogue rise time can be slower
    than one sampling interval. This method corrects for those
    scenarios.
    Can use gapless if pulse is uniform (e.g., frame pulses).
    Increase si, the gap factor, if digitizer is super duper
    fast.
    """
    idx = np.flatnonzero(np.diff(t) > thres)
    if gapless:
        si, _ = stats.mode(np.diff(idx))
        si = si / 2
    good = np.flatnonzero(np.diff(idx) > si)
    heads = idx[np.concatenate(([0], good + 1))]
    tails = idx[np.concatenate((good, [-1]))] + 1
    return heads, tails
