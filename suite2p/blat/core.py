import os
import re
import numpy as np
import warnings
from .behaviour import extract_behaviour, stitch, extract_plane
from .space import pc_analysis
from .bayes import crossvalidate
from .longitudinal import mkmask

default_ops = {
    'files': {
        'subfolder': 'suite2p',
    },
    'load': {
        'spks': 'spks.npy',
        'iscell': 'iscell.npy',
        'model': 'model.npy',
        'ops': 'ops.npy',
        'stat': 'stat.npy',
    },
    'space': {
        'length': 180.0,
        'bins': 80,
        'sigma': 4, # smoothing kernel in cm
        'nboots': 1_000, # number of permutations/bootstraps
        'alpha': .05,
    },
    'bayes': {
        'bins': 80,
        'sigma':4,
        'dt': 2,
        'k': 10,
    },
    'imaging': {
        'flybacks': None,
    },
}

class planepack():
    def __init__(self, spks, iscell, plane, behaviour, model, ops, my_ops, stat):
        self.behaviour = behaviour
        self.iscell = iscell
        self.spks = spks
        self.plane = plane
        self.behaviour = behaviour
        self.model = model
        self.ops = my_ops
        self.s2p_ops = ops
        self.stat = stat

        self.imaging()
        self.pc_analysis()
        self.decode()

    def imaging(self):
        ops = self.s2p_ops[()]

        Lx = ops['Lx']
        Ly = ops['Ly']
        
        xoff = np.concatenate(([0], ops['xoff']))
        yoff = np.concatenate(([0], ops['yoff']))
        self.regshift = np.sqrt(np.diff(xoff)**2 + np.diff(yoff)**2)
        
        green = ops['meanImg']
        green = (green - np.min(green)) / np.ptp(green)
        red = ops['meanImg_chan2']
        red = (red - np.min(red)) / np.ptp(red)
        self.mimg = np.stack((red, green, np.zeros(green.shape)), axis=2)
        
        iscell = np.flatnonzero(self.iscell)
        self.mask = mkmask(self.stat, Ly, Lx, iscell)

    
    def decode(self):
        print('running maximum a posteriori estimation')
        mvt = self.behaviour['movement']
        x = self.behaviour['position'][mvt]
        n = self.spks[:, mvt]
        n = n[~np.all(n == 0, axis=1), :]
        self.bayes = crossvalidate(x, n, bins=self.ops['bayes']['bins'], \
                            dt=np.round(self.ops['bayes']['dt'] * self.behaviour['fs']).astype(int), \
                            sigma=self.ops['bayes']['sigma'] * self.ops['bayes']['bins'] / self.ops['space']['length'], \
                            k=self.ops['bayes']['k'])


    def pc_analysis(self, permute=None):
        print('running place cells analysis for plane', self.plane)
        if hasattr(self, 'analysis'):
            if None not in self.analysis['tests']['SI']:
                ans = input('Permutation tests already performed. Run again? y/[n]\t')
                if ans == 'y':
                    permute = True
                elif (ans == 'n') | (ans == ''):
                    permute = False
                else:
                    raise ValueError('Bitch, wtf?')
        else:
            self.analysis = {}
            if permute is None:
                permute = True

        if permute:
            self.analysis = pc_analysis(self.behaviour, self.spks, bins=self.ops['space']['bins'], \
                                sigma=self.ops['space']['sigma'] * self.ops['space']['bins'] / self.ops['space']['length'], \
                                nboots=self.ops['space']['nboots'], alpha=self.ops['space']['alpha'])
        else:
            self.analysis.update(pc_analysis(self.behaviour, self.spks, bins=self.ops['space']['bins'], \
                                sigma=self.ops['space']['sigma'] * self.ops['space']['bins'] / self.ops['space']['length'], \
                                nboots=None, alpha=self.ops['space']['alpha']))


class blatify():
    def __init__(self, path, ops={}):
        self.mkops(ops)
        self.load(path)

    def mkops(self, ops={}):
        if not hasattr(self, 'ops'):
            self.ops = default_ops
        for key in ops.keys():
            if key in self.ops.keys():
                self.ops[key].update(ops[key])
            else:
                warnings.warn(key + ' is not recognized as a valid option.', RuntimeWarning)

    def load(self, path):
        files = os.listdir(path)
        beh_files = sorted([f for f in files if re.search('^behaviour_\d+.h5$', f)])
        print('found', len(beh_files), 'behaviour files:', beh_files)
        planes = os.listdir(os.path.join(path, 'suite2p'))
        planes = sorted([f for f in planes if re.search('^plane\d+$', f)])
        print('found', len(planes), 'planes:', planes)

        if self.ops['imaging']['flybacks'] is None:
            flybacks = [False] * len(planes)
        else:
            flybacks = self.ops['imaging']['flybacks']
        if len(flybacks) != len(planes):
            raise RuntimeError('Length of flybacks list does not match number of planes.')

        behaviours = []
        for beh in beh_files:
            print('extracting behaviour', beh)
            behaviours.append(extract_behaviour(os.path.join(path, beh), normalize=self.ops['space']['length']))
        behaviour = stitch(behaviours)

        self.plane = []
        for (i, plane), fly in zip(enumerate(planes), flybacks):
            if not fly:
                print('loading', plane)
                data = self.ops['load'].copy()
                for key, file in data.items():
                    data[key] = np.load(os.path.join(path, self.ops['files']['subfolder'], plane, file), allow_pickle=True)
                data['iscell'] = data['iscell'][:, 0].astype(bool)
                data['spks'] = data['spks'][data['iscell'], :]
                data['plane'] = i
                data['behaviour'] = extract_plane(behaviour, plane=i, nplanes=len(planes))
                data['my_ops'] = self.ops
                self.plane.append(planepack(**data))

        ops = {
            'files': {
                'root': path,
                'behaviour': beh_files,
                'planes': planes,
            },
            'imaging': {
                'overall_fs': behaviour['fs'],
                'nplanes': len(planes),
                'flybacks': flybacks,
            },
        }
        self.mkops(ops)
