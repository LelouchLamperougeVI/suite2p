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