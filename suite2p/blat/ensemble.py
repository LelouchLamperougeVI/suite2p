import numpy as np
from suite2p.blat import core, utils, quaser, space
from scipy import stats
import warnings

class ensembles:
    def __init__(self,
                 plane: core.planepack,
                 w: float = 1.,
                 alpha: float = .05,
                 fdr: bool = True,
                 min_size: int = 10,
        ):
        '''
        Detect resting state ensembles. Featuring the new QuaSer method.

        Inputs
        ======
        plane (planepack)
            planepack object from core pipeline
        w (float) --- 1 s
            window size used for decimating deconvolved trains
        alpha (float) --- .05
            alpha level for building association/boolean matrix
        fdr (bool) --- True
            whether or not to apply Benjamini-Hochberg correction
        min_size (int) --- 10
            minimum of size of ensembles to be considered

        Properties
        ==========
        ev (list[float])
            explained variance (ev[0]) and reverse explained variance (ev[1])
        rest1/rest2 (dict)
            key analysis results for rest 1 and rest 2 epochs, organized as:
            {
                'ensembles': list[list[int]], # ensemble member indices
                'weights': np.ndarray, # ensemble neuron weights, neurons x ensembles
                'activation': np.ndarray, # ensemble time-courses, ensembles x time
                'order': list[int], # optimal matrix ordering from QuaSer TSP seriation
                'analysis': dict, # ensemble activation during RUN, analyzed through pc_analysis
            }
        '''
        fs = plane.ops['imaging']['fs']
        w = fs * w
        self.sigma = np.round(w / 2).astype(int) # sigma is half of w
        self.w = np.round(w).astype(int)

        self.behaviour = plane.behaviour.copy()
        self.spks = plane.spks.copy()
        self.fdr = fdr
        self.alpha = alpha
        self.min_size = min_size

        self.pc_order = np.argsort(np.argmax(plane.analysis['smooth']['stack'], axis=1))
        self.space_ops = plane.ops['space']
        self.stack = plane.analysis['smooth']['stack']
        self.rasters = plane.analysis['smooth']['rasters']
        self.ispc = plane.analysis['ispc']

        self._mk_epochs()
        if np.all(~self.epochs[0]) | np.all(~self.epochs[2]):
            warnings.warn("no immobile epochs detected, returning None")
            return None
        self._calc_ev()
        self._detect()

    def _mk_epochs(self):
        '''
        Partition epochs into rest1, run and rest2.
        Remove moving epochs for rests and only consider moving epochs for run.
        '''
        mvt = self.behaviour['movement']
        epoch = self.behaviour['epochs']
        rest = np.cumsum(np.abs(np.diff(epoch == 1)))
        rest *= rest % 2
        rest[rest == 3] = 2
        rest = np.concatenate(([0], rest))
        rest1 = (epoch == 1) & ~mvt & (rest == 1)
        run = (epoch > 1) & mvt
        rest2 = (epoch == 1) & ~mvt & (rest == 2)
        self.epochs = [rest1, run, rest2]

    def _corr_mat(self, epoch: int):
        '''
        Compute correlation matrix for an epoch after smoothing q-matrix.
        '''
        spks = self.spks[:, self.epochs[epoch]].copy()
        spks = utils.fast_smooth(spks, sigma=self.sigma, axis=1)
        # plt.figure()
        # plt.imshow(stats.zscore(spks[self.pc_order, :], axis=1), aspect='auto', interpolation='none')
        return utils.corr(spks)

    def _calc_ev(self):
        '''
        Calculate explained variance and reverse explained variance.
        '''
        r = [self._corr_mat(e) for e in range(len(self.epochs))]
        idx = np.triu(np.ones_like(r[0], dtype=bool), 1)
        corr = lambda i, j: utils.corr(r[i][idx], r[j][idx])
        rho_xy = corr(1, 2) # x: rest2, y: run, z: rest1
        rho_xz = corr(0, 2)
        rho_yz = corr(0, 1)

        ev = (rho_xy - rho_xz * rho_yz) / np.sqrt(1 - rho_xz**2) / np.sqrt(1 - rho_yz**2)
        rev = (rho_yz - rho_xz * rho_xy) / np.sqrt(1 - rho_xz**2) / np.sqrt(1 - rho_xy**2)
        self.ev = [ev, rev]

    def _asso(self, epoch: int):
        '''
        Create boolean association matrix for cells pairs at specified epoch.
        Principle:
            Let X ~ Poisson(l_x) and Y ~ Poisson(l_y). Let Z = X & Y be a new
            process that tracks the co-occurance of X = 1 and Y = 1. Then,
            the expected probability of Z is l_x * l_y. Compare the observed
            value against the theoretical in a poisson distribution to obtain
            p-value. To combat temporal jitter, reduce X and Y by checking if
            a spike occured within consecutive windows of size w. Use the
            Benjamini–Hochberg procedure at specified alpha level to detect
            associations.
        '''
        w = self.w
        spks = self.spks[:, self.epochs[epoch]].copy()

        p = np.zeros((spks.shape[0],) * 2)
        for i, a in enumerate(spks):
            a = np.convolve(a > 0, np.ones(w), mode='valid') > 0
            a = a[::w]
            for j, b in enumerate(spks):
                b = np.convolve(b > 0, np.ones(w), mode='valid') > 0
                b = b[::w]
                theo = len(a) * np.mean(a) * np.mean(b)
                if theo == 0.:
                    p[i, j] = 1
                    continue
                obs = np.sum(a & b)
                p[i, j] = stats.poisson.sf(obs, theo)
        p = (p + p.T) / 2

        if self.fdr:
            fdr = p[np.triu(np.ones_like(p, dtype=bool), 1)]
            fdr = np.sort(fdr)
            fdr = fdr[fdr <= ((np.arange(len(fdr)) + 1) * self.alpha / len(fdr))]
            if len(fdr) == 0:
                return np.eye(len(p), dtype=bool)
            fdr = np.max(fdr)
        else:
            fdr = self.alpha

        return p <= fdr

    def _nmf(self,
             epoch: int | str,
             communities = None,
             W = None,
             H = None,
             smooth=True,
             tol: float = 1e-4,
             niter: int = 1_000
        ):
        '''
        Extract ensemble neuron weights (W) and reactivation
        time-courses (H) using multiplicative update rules.

        To fix the weights of one matrix, provide either W or H.
        '''
        assert communities is not None or W is not None, "either communities or W must be provided"
        if W is not None and H is not None:
            return W, H

        W_frozen = False
        H_frozen = False

        if W is None:
            W = np.zeros((self.spks.shape[0], len(communities)))
            for i, c in enumerate(communities):
                W[c, i] = 1.
        else:
            W_frozen = True

        if epoch == 'full':
            V = self.spks
        else:
            V = self.spks[:, self.epochs[epoch]]
        if smooth:
            V = utils.fast_smooth(V, self.sigma, axis=1)
        V /= np.std(V, axis=1, keepdims=True)
        V[np.isnan(V)] = 0

        if H is None:
            H = W.T @ V
        else:
            H_frozen = True

        e = np.inf
        for _ in range(niter):
            if not H_frozen:
                H = H * (W.T @ V) / (W.T @ W @ H)
                H[np.isnan(H)] = 0
            if not W_frozen:
                W = W * (V @ H.T) / (W @ H @ H.T)
                W[np.isnan(W)] = 0
            if (e - (e := np.linalg.norm(V - W @ H, 'fro'))) < tol:
                break

        return W, H

    def _detect(self):
        '''
        Detect ensembles using QuaSer.
        '''
        run = self._asso(1)
        p = np.mean(run[np.triu(np.ones_like(run, dtype=bool), 1)])

        def get_ensembles(epoch):
            md = quaser.BMF(self._asso(epoch))
            communities = [c for c in md.communities if len(c) >= self.min_size]
            ret = []
            for c in communities:
                k = run[np.ix_(c, c)]
                k = k[np.triu(np.ones_like(k, dtype=bool), 1)]
                n = len(k)
                k = np.sum(k)
                pval = stats.binom.sf(k, n, p)
                if pval <= self.alpha:
                    ret.append(c)

            W, H = self._nmf(epoch=epoch, communities=ret)
            _, H_ = self._nmf(epoch='full', W=W, smooth=False)
            if H_.shape[0] > 0:
                analysis = space.pc_analysis(behaviour=self.behaviour,
                                              spks=H_,
                                              bins=self.space_ops['bins'],
                                              sigma=self.space_ops['sigma'],
                                              alpha=self.space_ops['alpha'],
                                              prctile=self.space_ops['percentile'],
                                              ksg_sigma=self.space_ops['ksg_sigma'],
                                              nboots=None,
                                             )
            else:
                analysis = None

            stacks = [self.stack[c, :] for c in communities]
            rasters = [self.rasters[c, :, :] for c in communities]
            pc_binom = [stats.binom.sf(np.sum(self.ispc[c]), len(c), np.mean(self.ispc)) for c in communities]
            ret = {
                'ensembles': ret, # communities
                'weights': W, # ensemble neuron weights
                'activation': H, # ensemble time-courses
                'order': md.order, # optimal matrix ordering from QuaSer TSP seriation
                'analysis': analysis, # run the ensemble activation through pc_analysis
                'stacks': stacks, # trajectory stacks
                'rasters': rasters, #
                'pc_binom': pc_binom, #
            }
            return ret

        self.rest1 = get_ensembles(0)
        self.rest2 = get_ensembles(2)

