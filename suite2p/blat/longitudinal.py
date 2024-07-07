import numpy as np
from skimage.transform import warp_polar, rotate
from scipy.signal import fftconvolve

def mkmask(stat, ops, iscell=None):
    """
    Create masks from suite2p stat and ops dicts
    for the cells defined in array iscell (bool or int).
    """
    if iscell is None:
        iscell = np.ones((stat.shape[0])).astype(bool)
    if iscell.dtype is np.dtype('bool'):
        iscell = np.flatnonzero(iscell)

    Lx = ops['Lx']
    Ly = ops['Ly']

    count = 0
    mask = -np.ones((Ly, Lx))
    for i, s in enumerate(stat):
        if i in iscell:
            for x, y in zip(s['xpix'], s['ypix']):
                mask[y, x] = count
            count += 1

    return mask


def lintransform(stat, Ly, Lx, rot=0, drifty=0, driftx=0):
    rot = -rot
    ret = stat.copy()
    origin = [Ly / 2, Lx / 2]
    for i in range(stat.shape[0]):
        ypix = stat[i]['ypix'] - origin[0]
        xpix = stat[i]['xpix'] - origin[1]
        ret[i]['ypix'] = np.round(ypix * np.cos(np.deg2rad(rot)) + xpix * np.sin(np.deg2rad(rot)) \
                                  + drifty + origin[0]).astype(int)
        ret[i]['xpix'] = np.round(xpix * np.cos(np.deg2rad(rot)) - ypix * np.sin(np.deg2rad(rot)) \
                                  + driftx + origin[1]).astype(int)
    return ret


def transform(mask, rot=0, drifty=0, driftx=0):
    reg = rotate(mask, rot, preserve_range=True, order=0, cval=-1)
    reg = np.roll(reg, drifty, axis=0)
    reg = np.roll(reg, driftx, axis=1)
    return reg


def regmasks(fixed, moving, twostep=True, maxIter=100):
    """
    A very unsophisticated registration algo for masks across days
    that just works... Register twice with twostep (recommended).
    """
    def do(moving):
        before = fixed
        reg = moving
        it = 0
        rot, drifty, driftx = 0, 0, 0
        while ~np.all(reg == before) & (it < maxIter):
            before = reg
            reg, r = polar_reg(fixed, reg)
            rot += r
            reg, y, x = trans_reg(fixed, reg)
            drifty += y
            driftx += x
            it += 1
        
        if it == maxIter:
            print('Maximum iterations reached. Registration did not converge.')
        else:
            print('Registration converged after', it, 'steps.')

        reg = transform(moving, rot=rot, drifty=drifty, driftx=driftx)
        return reg, rot, drifty, driftx

    reg, rot, drifty, driftx = do(moving)
    if twostep:
        print('Two-step enabled. Registering a second time.')
        reg, crot, cdrifty, cdriftx = do(reg)
        rot += crot
        drifty += cdrifty
        driftx += cdriftx

    return reg, rot, drifty, driftx


def trans_reg(fixed, moving, shift=.2):
    """
    Register translated masks, with maximum shift in fraction of resolution.
    """
    M = fixed.shape[0]
    N = fixed.shape[1]
    shifty = np.round(M * shift).astype(int)
    shiftx = np.round(N * shift).astype(int)
    
    corr = fftconvolve((fixed+1).astype(bool), np.flip((moving+1).astype(bool)))
    corr = corr[(M - shifty):(M + shifty + 1), :]
    corr = corr[:, (N - shiftx):(N + shiftx + 1)]
    
    drifty, driftx = np.unravel_index(np.argmax(corr), corr.shape)
    drifty -= shifty + 1
    driftx -= shiftx + 1

    moving = transform(moving, drifty=drifty, driftx=driftx)
    return moving, drifty, driftx


def polar_reg(fixed, moving, shift=15):
    """
    Register rotated masks, with maximum rotation shift in degrees.
    """
    radius = np.floor(np.min(fixed.shape) / 2)
    warped_fixed = warp_polar(fixed+1, radius=radius, scaling="log").astype(bool)
    warped_moving = warp_polar(moving+1, radius=radius, scaling="log").astype(bool)
    
    scale = np.arange(1, radius+1) / radius
    rot = np.arange(-shift, shift+1)
    score = np.empty_like(rot)
    for i in range(rot.shape[0]):
        score[i] = np.sum(np.sum(warped_fixed * np.roll(warped_moving, \
                                         shift=rot[i], axis=0), axis=0) * scale)

    rot = -rot[np.argmax(score)]
    moving = transform(moving, rot=rot)
    return moving, rot