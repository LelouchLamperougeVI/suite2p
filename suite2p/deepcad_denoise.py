from deepcad.test_collection import testing_class
from suite2p import io
import yaml
import os
import numpy as np
import tifffile
import torch

def deepcad_denoise(binary, md_path, batch=10_000, tmp_path='/var/tmp/deepcad', GPU='0'):
    """
    Integration of DeepCAD-RT into the suite2p pipeline
    """
    defaults = {
        'test_datasize': 1_000_000_000, # number of frames, do them all
        'GPU': GPU,
        'visualize_images_per_epoch': False,
        'display_images': False,
        'pth_dir': os.path.abspath(os.path.join(md_path, os.pardir)),
        'denoise_model': os.path.basename(os.path.normpath(md_path)),
        'output_dir': os.path.join(tmp_path, 'results'),
        'datasets_path': os.path.join(tmp_path, 'data'),
    }
    # Since the receptive field of 3D-Unet is ~90, seamless stitching requires an overlap (patch_xyt*overlap_factor）of at least 90 pixels.

    with open(os.path.join(md_path, 'para.yaml')) as stream:
        params = yaml.safe_load(stream)
    params.update(defaults)

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])
    if not os.path.exists(params['datasets_path']):
        os.makedirs(params['datasets_path'])

    n_frames = binary.shape[0]
    for i in range(int(np.ceil(n_frames / batch))):
        frame_range = {
            'frame_range': [batch * i, np.min([batch * (i + 1), n_frames])],
        }
        binary.write_tiff(os.path.join(params['datasets_path'], 'temp{:06d}.tif'.format(i)), frame_range)

    tc = testing_class(params)
    tc.run()

    out_path = os.path.join(tc.output_path, os.path.splitext(tc.model_list)[0])
    out = [f for f in os.listdir(out_path) if f.endswith('.tif')]
    out = [os.path.join(out_path, f) for f in out]
    out.sort()
    accum = 0
    for f in out:
        stack = tifffile.imread(f)
        binary[accum : accum + stack.shape[0], :, :] = stack
        accum += stack.shape[0]

    for root, dirs, files in os.walk(tmp_path, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))

    torch.cuda.empty_cache()