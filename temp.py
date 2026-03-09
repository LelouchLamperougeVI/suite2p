import numpy as np
import sys
import suite2p

db = [
    {
    'data_path': ['/mnt/DATA/mini2p/2025_11_25'],
    'subfolders': [],
    },
]


ops = suite2p.default_ops()

# recording-specific settings
ops['nplanes'] = 1
ops['nchannels'] = 1
ops['functional_chan'] = 1
ops['deconv_tau'] = 10 # tau for oasis deconvolution
ops['tau'] = 1.0
ops['fs'] = 5
ops['do_registration'] = True
ops['align_by_chan'] = 1
ops['combined'] = False

# registration parameters
ops['nimg_init'] = 500
ops['batch_size'] = 500 # increase if your computer is not a piece of shit
ops['smooth_sigma'] = 1.15
ops['smooth_sigma_time'] = 1.0
ops['spatial_scale'] = 0
ops['nonrigid'] = True

# deepcad denoising
ops['deepcad'] = '/home/loulou/Documents/GitHub/suite2p/deepcad_md'
ops['denoised'] = False

# ROI detection
ops['do_registration'] = 1 # force registration when > 1
ops['anatomical_only'] = False
ops['denoise'] = True
ops['sparse_mode'] = True
ops['nbinned'] = 50_000
ops['threshold_scaling'] = .75
ops['max_iterations'] = 30
ops['high_pass'] = 200
ops['spatial_hp_detect'] = 75
ops['active_percentile'] = 0

# neuropil settings
ops["allow_overlap"] = True # for dense populations with overlapping neurons
ops['inner_neuropil_radius'] = 1
ops['outer_neuropil_radius'] = 8
ops['neucoeff'] = 1.0
ops['baseline'] = 'constant_percentile'
ops['prctile_baseline'] = 10.0


for d in db:
    suite2p.run_s2p(ops=ops, db=d)
