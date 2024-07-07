import numpy as np
import sys
import suite2p

db = [{
    'data_path': ['/mnt/DATA/CA3/Bernard/2024_07_02'],
    'subfolders': [],
}]


ops = suite2p.default_ops()

ops['nplanes'] = 4
ops['nchannels'] = 2
ops['functional_chan'] = 1
ops['tau'] = 10.0
ops['fs'] = 7.5
ops['do_registration'] = True
ops['align_by_chan'] = 1
ops['nimg_init'] = 500
#ops['batch_size'] = 1_000
ops['batch_size'] = 500
ops['smooth_sigma'] = 1.15
ops['smooth_sigma_time'] = 0.0
ops['two_step_registration'] = False

ops['inner_neuropil_radius'] = 1
ops['outer_neuropil_radius'] = 8

ops['neucoeff'] = 1.0

ops['baseline'] = 'constant_percentile'
ops['prctile_baseline'] = 10.0

for d in db:
    suite2p.run_s2p(ops=ops, db=d)
