from suite2p import io
import numpy as np

fn = '/mnt/DATA/CA3/Donald/2025_11_25/suite2p/plane1/data.bin'
with io.BinaryFile(Ly=308, Lx=512, filename=fn, n_frames=14800) as f_reg:
    print(f_reg.shape)
    frame = f_reg[1110, :, :]

print(frame)
