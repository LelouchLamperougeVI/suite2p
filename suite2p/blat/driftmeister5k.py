import numpy as np
import cv2
from ScanImageTiffReader import ScanImageTiffReader
from suite2p.blat import utils
from scipy.interpolate import make_smoothing_spline

def _register(im, xoff, yoff, crop):
    im = np.roll(im, -yoff, axis=0)
    im = np.roll(im, -xoff, axis=1)
    im = im[crop[0]:-crop[1], :]
    im = im[:, crop[2]:-crop[3]]
    return im

def _locate_ref(sm, idx, wdw):
    ref = utils.corr(sm[idx, :, :].flatten(), sm.reshape((sm.shape[0], -1)))
    ref = ref.squeeze()
    x = np.arange(len(ref))
    spl = make_smoothing_spline(x[::wdw], ref[::wdw], lam=1e12)
    ref[np.max([0, idx - wdw]):idx] = np.nan
    ref[idx:np.min([len(ref), idx + wdw + 1])] = np.nan
    # return np.argmin(spl(x)), spl(x)
    return np.argmin(spl(x)), ref

def zmotion(tif_file, ops_file, wdw=100, decimate=4, ndepths=10):
    '''
    Estimate Z-motion drift from single plane imaging data.

    Arguments
    =========
        tif_file:    tiff file path
        ops_file:    ops file path
        wdw:         number smoothing frames
        decimate:    downsampling factor
        ndepths:     number of estimation depth planes

    Output
    ======
        drift:       inferred Z-motion drift time vector in units of normalized
                     Pearson correlation (A.U.)

    TODO: choose plane number in tif
    '''

    # load up registration data from suite2p
    ops = np.load(ops_file, allow_pickle=True)
    ops = ops[()]
    xoff = ops['xoff']
    yoff = ops['yoff']

    crop = [-np.min(yoff), np.max(yoff), -np.min(xoff), np.max(xoff)]

    # load up tiff and downsample image by bicubic interp
    with ScanImageTiffReader(tif_file) as tif:
        sz = tif.shape()
        sz[1] -= crop[0] + crop[1]
        sz[2] -= crop[2] + crop[3]
        x = sz[1] // decimate
        y = sz[2] // decimate
        mimg = np.empty((sz[0], x, y))
        for i in range(sz[0]):
            im = np.squeeze(tif.data(beg=i, end=i+1))
            im = _register(im, xoff[i], yoff[i], crop)
            mimg[i, :, :] = cv2.resize(im, (y, x), interpolation=cv2.INTER_CUBIC)

    # smooth pixels over time
    kernel = np.ones((wdw,)) / wdw
    norm = np.convolve(np.ones((mimg.shape[0])), kernel, mode='same')
    sm = np.empty_like(mimg)
    for i in range(sm.shape[1]):
        for j in range(sm.shape[2]):
            sm[:, i, j] = np.convolve(mimg[:, i, j], kernel, mode='same') / norm

    # locate top and bottom reference planes
    fuzzyD = []
    x, _ = _locate_ref(sm, sm.shape[0] // 2, wdw)
    x, ref = _locate_ref(sm, x, wdw)
    ref  = ref - np.nanmean(ref)
    fuzzyD.append(ref)
    _, ref = _locate_ref(sm, x, wdw)
    ref = np.nanmean(ref) - ref
    fuzzyD.append(ref)

    # locate intermediate planes
    est = np.nanmean(fuzzyD, axis=0)
    x = np.arange(len(est))
    spl = make_smoothing_spline(x[::wdw], est[::wdw], lam=1e12)
    est = spl(x)
    depths = np.linspace(np.min(est), np.max(est), ndepths)

    for d in depths[1:-1]:
        idx = np.argmin(np.abs(d - est))
        _, ref = _locate_ref(sm, idx, wdw)
        if idx - wdw - 1 < 0:
            inflection = ref[idx + wdw + 1]
        elif idx + wdw + 1 < 0:
            inflection = ref[idx - wdw - 1]
        else:
            inflection = (ref[idx + wdw + 1] + ref[idx - wdw - 1]) / 2
        ref[:idx] = 2 * inflection - ref[:idx]
        if utils.corr(ref, est).squeeze() < 0:
            ref = -ref
        fuzzyD.append(ref)

    fuzzyD = np.array(fuzzyD)
    fuzzyD = fuzzyD - np.nanmean(fuzzyD, axis=1, keepdims=True)

    drift = np.nanmedian(fuzzyD, axis=0)
    return drift
