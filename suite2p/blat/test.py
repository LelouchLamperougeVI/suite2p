from ctypes import *
import numpy as np

def twocol_unique(x: np.ndarray):
    ctwocol_unique = CDLL('/home/loulou/Documents/GitHub/suite2p/suite2p/blat/twocol_unique.so').twocol_unique
    # ctwocol_unique.restype = POINTER(c_double * x.size)
    x = x.flatten()
    ret = ctwocol_unique(x.ctypes.data_as(POINTER(c_double)), int(x.size/2))
    # ret = np.ctypeslib.as_array(ret.contents, shape=(x.size,))
    idx = np.flatnonzero(np.isnan(x))[0]
    print(idx)
    
    x = x[:idx].reshape(-1, 2)
    
    return x

x = np.array([0, 0, 6, 9, 6, 9, 0, 0, 1, 0, 0, 1], dtype=np.float64).reshape(-1, 2)
x = twocol_unique(x)
print(x)






import matplotlib.pyplot as plt
from suite2p.blat.longitudinal import *

def jaccard(fixed, moving, thres=.25):
    idx = np.max(moving).astype(int) + 1
    d = np.zeros((idx,))
    match = np.empty((idx,))
    match.fill(np.nan)
    for i in range(idx):
        mask = moving == i
        union = fixed[mask]
        if np.all(union == -1):
            continue
        candidates, counts = np.unique(union[union > -1], return_counts=True)
        union = [c / ( np.sum(fixed == u) + np.sum(mask) - c ) for u, c in zip(candidates, counts)]
        idx = np.argmax(union)
        d[i] = union[idx]
        if d[i] > thres:
            match[i] = candidates[idx]

    return match, d

def jaccard(fixed, moving, thres=.25):
    d = np.zeros(moving.shape)
    match = np.empty(moving.shape)
    match.fill(np.nan)
    candidates = [[f['ypix'], f['xpix']] for f in fixed]
    for i in range(moving.shape[0]):
        coor = [moving[i]['ypix'], moving[i]['xpix']]
        for c in candidates:
            idx = np.isin(coor[0], c[0]) & np.isin(coor[1], c[1])

stat = np.load('/mnt/DATA/CA3/Bernard/2024_06_26/suite2p/plane3/stat.npy', allow_pickle=True)
ops = np.load('/mnt/DATA/CA3/Bernard/2024_06_26/suite2p/plane3/ops.npy', allow_pickle=True)
ops = ops[()]
iscell = np.load('/mnt/DATA/CA3/Bernard/2024_06_26/suite2p/plane3/iscell.npy', allow_pickle=True)
iscell = iscell[:, 0].astype(bool)

fixed = mkmask(stat, ops, iscell)

stat = np.load('/mnt/DATA/CA3/Bernard/2024_06_27/suite2p/plane3/stat.npy', allow_pickle=True)
ops = np.load('/mnt/DATA/CA3/Bernard/2024_06_27/suite2p/plane3/ops.npy', allow_pickle=True)
ops = ops[()]
iscell = np.load('/mnt/DATA/CA3/Bernard/2024_06_27/suite2p/plane3/iscell.npy', allow_pickle=True)
iscell = iscell[:, 0].astype(bool)

moving = mkmask(stat, ops, iscell)

# test = polar_reg(fixed, moving)
test, rot, drifty, driftx = regmasks(fixed, moving)
%lprun -f jaccard match, d = jaccard(fixed, test)
match = match[d > .25]

test = lintransform(stat, ops['Ly'], ops['Lx'], rot, drifty, driftx)
test = mkmask(test, ops, iscell)

fig, axs = plt.subplots(1, 2)
axs[0].imshow((fixed+1).astype(bool).astype(int) + ((moving+1).astype(bool).astype(int) *2))
axs[1].imshow((fixed+1).astype(bool).astype(int) + ((test+1).astype(bool).astype(int) *2))

fig = plt.figure()
plt.imshow(np.isin(fixed, match) + ((test+1).astype(bool).astype(int) *2))