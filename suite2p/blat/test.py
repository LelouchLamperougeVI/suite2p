from ctypes import *
import numpy as np

twocol_unique = CDLL('/home/loulou/Documents/pysandbox/a.out').twocol_unique

test = np.array([0, 0, 6, 9, 6, 9, 0, 0, 1, 0, 0, 1], dtype=np.float64)
twocol_unique.restype = POINTER(c_double * test.shape[0])
ret = twocol_unique(test.ctypes.data_as(POINTER(c_double)), 6)
ret = np.ctypeslib.as_array(ret.contents, shape=(6,))
print(ret)
print(ret.dtype)
