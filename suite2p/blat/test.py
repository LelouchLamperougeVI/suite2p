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
from suite2p.blat.longitudinal import crossdays
from suite2p.blat import blatify

ops = {
    'imaging': {
        'flybacks': [True, False] * 2,
    },
}

a = blatify('/home/loulou/Documents/Data/CA3/Bernard/2024_06_26', ops)
b = blatify('/home/loulou/Documents/Data/CA3/Bernard/2024_06_27', ops)
c = blatify('/home/loulou/Documents/Data/CA3/Bernard/2024_06_28', ops)



reg = crossdays([a.plane[1], b.plane[1], c.plane[1]])



stack = a.plane[1].analysis['smooth']['stack'].T
stack = (stack - np.min(stack, axis=0)) / np.ptp(stack, axis=0)
order = np.argsort(np.argmax(stack, axis=0))

fig, axs = plt.subplots(1, 3)
stack_a = stack[:, order]
axs[0].imshow(stack_a.T)

stack = b.plane[1].analysis['smooth']['stack'].T
stack = (stack - np.min(stack, axis=0)) / np.ptp(stack, axis=0)
idx = reg[1][0][order]
sstack = np.zeros((stack.shape[0], idx.shape[0]))
sstack[:, ~np.isnan(idx)] = stack[:, idx[~np.isnan(idx)].astype(int)]
stack_b = sstack
axs[1].imshow(stack_b.T)

stack = c.plane[1].analysis['smooth']['stack'].T
stack = (stack - np.min(stack, axis=0)) / np.ptp(stack, axis=0)
idx = reg[2][0][order]
sstack = np.zeros((stack.shape[0], idx.shape[0]))
sstack[:, ~np.isnan(idx)] = stack[:, idx[~np.isnan(idx)].astype(int)]
stack_c = sstack
axs[2].imshow(stack_c.T)




from suite2p.blat import utils

fig, axs = plt.subplots(1, 2)
axs[0].imshow(utils.corr(stack_a, stack_b), vmin=0)
axs[1].imshow(utils.corr(stack_a, stack_c), vmin=0)