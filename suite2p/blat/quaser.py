import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

def lk_tsp(dist, max_cand=5):
    n = dist.shape[0]
    cand = [
        sorted(range(n), key=lambda j: dist[i][j])[1 : max_cand + 1]
        for i in range(n)
    ]

    # Nearest Neighbor tour initialization
    tour, unvisited = [0], set(range(1, n))
    while unvisited:
        tour.append(min(unvisited, key=lambda x: dist[tour[-1]][x]))
        unvisited.remove(tour[-1])

    # Dynamic depth LK search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            t1, t2 = tour[i], tour[(i + 1) % n]
            for t3 in cand[t2]:
                if t3 == t1:
                    continue
                g1 = dist[t1][t2] - dist[t2][t3]
                if g1 <= 1e-6:
                    continue

                idx3 = tour.index(t3)
                t4 = tour[(idx3 - 1) % n]
                if g1 + dist[t3][t4] - dist[t4][t1] > 1e-6:
                    i1, i3 = sorted([(i + 1) % n, idx3])
                    tour[i1:i3] = reversed(tour[i1:i3])
                    improved = True
                    break
            if improved:
                break

    return tour


def jaccard(B: np.ndarray) -> np.ndarray:
    '''
    Compute Jaccard distance matrix.
    '''
    B_int = B.astype(np.int32)
    intersection = B_int @ B_int.T
    degrees = B_int.sum(axis=1)
    union = degrees[:, None] + degrees[None, :] - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        J = np.true_divide(intersection, union)
        J[union == 0] = 0.0  # Handle 0/0 division for isolated nodes
        
    return 1.0 - J

def nn2opt(D: np.ndarray, s: int=None, opt=True) -> list[int]:
    '''
    Nearest neighbour heuristic solution with 2-opt
    optimization to the travelling salesman problem.

    ===Inputs===
    D (ndarray)    Symmetric distance matrix
    s (int)        Starting vertex. If None, use
                   node with lowest degree.
    opt (bool)     Whether or not to use 2-opt

    ===Outputs===
    order (list)   Order of route
    '''
    assert np.allclose(D, D.T), 'D must be symmetric'
    
    if D.shape[0] <= 1:
        return list(range(n))
    if s is None:
        s = int(np.argmax(np.sum(D, axis=0)))
    
    D_ = D.copy()
    # greedy search
    D_[np.diag(np.ones((D_.shape[0],), dtype=bool))] = np.inf
    order = [s,]
    for _ in range(D_.shape[0] - 1):
        order.append(np.argmin(D_[order[-1], :]))
        D_[order[-2], :] = np.inf
        D_[:, order[-2]] = np.inf

    # 2-opt
    if opt:
        improved = True
        while improved:
            improved = False
            for i in range(1, len(order) - 2):
                for j in range(i + 1, len(order) - 1):
                    d = -D[order[i-1], order[i]] - D[order[j], order[j+1]] \
                        + D[order[i-1], order[j]] + D[order[i], order[j+1]]
                    if d < -1e-8:
                        order[i:j+1] = order[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break

    return order

def sort(B: np.ndarray) -> list[int]:
    '''
    Sort binary matrix to closely satisfy C1P.
    '''
    return nn2opt(jaccard(B))
    #return lk_tsp(jaccard(B))

def mdl(B, pat = None):
    '''
    Minimum description length
    '''
    def entropy(p):
        p = np.clip(p, 1e-12, 1. - 1e-12)
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    if pat is not None:
        if len(pat.shape) == 1:
            pat = pat[np.newaxis, :]
        k, n = pat.shape
    
        p = np.mean(pat, axis=1)
        L = k * np.log2(n + 1) + n * np.sum(entropy(p))

        N = n * (n + 1) / 2
        err = B ^ ((pat.T @ pat) > 0)
        p = np.sum(np.triu(err)) / N
        L += np.log2(N + 1) + N * entropy(p)
    else:
        n = B.shape[0]
        N = n * (n + 1) / 2
        p = np.sum(np.triu(B)) / N
        L = np.log2(N + 1) + N * entropy(p)
    return L

def bishrinkgrow(B, pat):
    '''
    Optimize patterns by minimizing MDL with greedy bidirectional shrink/growth
    '''
    def step(i, ind, direction, val, opt):
        ind += direction
        while ind >= 0 and ind < pat.shape[1]:
            pat[i, ind] = val
            if (step_opt := mdl(B, pat)) > opt:
                pat[i, ind] = not val
                break
            opt = step_opt
            ind += direction
        return opt, ind

    for i, r in enumerate(pat):
        if (seg := np.flatnonzero(r)).size:
            cuts = np.flatnonzero(np.diff(seg) > 1) + 1
            seg = np.split(seg, cuts)
            for s in seg:
                left, right = s[0], s[-1]
                opt = mdl(B, pat)
                
                opt, left = step(i, left, -1, True, opt)
                opt, _ = step(i, left, 1, False, opt)
                opt, right = step(i, right, 1, True, opt)
                opt, _ = step(i, right, -1, False, opt)

    return pat

def quasi_cliques(B, pats, noise):
    '''
    Extract quasi-cliques from patterns.
    Greedily eliminate the least connected vertices
    until noise floor is reached.
    '''
    if np.isnan(noise):
        noise = 0
    R = np.zeros_like(B, dtype=bool)
    for i, r in enumerate(pats):
        idx = np.flatnonzero(r)
        while len(idx) > 0:
            residual = B[np.ix_(idx, idx)]
            if np.mean(~residual) <= noise:
                break
            idx = np.setdiff1d(idx, idx[np.argsort(np.sum(residual, axis=0))[0]])
        R[i, idx] = True

    return R

    
@dataclass
class BMF:
    B: np.ndarray
    communities: list[list[int]] = field(default_factory=list)
    order: list[int] | np.ndarray = field(default_factory=list)
    inverse_order: list[int] | np.ndarray = field(default_factory=list)
    _quasi: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _detected: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _refined: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _filled: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    opt_mdl: float = 0.
    
    def __post_init__(self):
        B = self.B.copy()
        if B.shape[0] != B.shape[1]:
            raise ValueError("B must be a square")
        if not np.allclose(B, B.T):
            raise ValueError("B must be symmetric")
        if B.dtype != bool:
            raise TypeError("B must be a boolean matrix")

        # 1. sort matrix by solving TSP
        order = sort(B)
        B = B[np.ix_(order, order)]
        self.B = B
        self.order = order
        self.inverse_order = np.argsort(order)
        
        # 2. minimize minimum description length
        R = B.copy()
        
        # filter out patterns that do not improve heuristic
        spacing = np.linspace(0, R.shape[0] - 1, np.round(np.mean(R) * R.shape[0]).astype(int))
        uni = np.zeros((1, R.shape[0]), dtype=bool)
        uni[0, np.round(spacing).astype(int)] = True
        thres = mdl(B, uni)
        pats = np.array([mdl(B, R[[i], :]) for i in range(R.shape[0])]) < thres
        if np.sum(pats) == 0:
            noise = np.mean(R)
        else:
            noise = R[~pats, :]
            noise = np.mean(noise)
            R = R[pats, :]
        
        # discard patterns that do not form quasi-cliques
        R = quasi_cliques(B, R, noise)
        self._quasi = R.copy()

        fill_gaps = lambda x: np.array([np.convolve(r, np.ones(3), mode='same') >= 2 for r in x])
        R = fill_gaps(R)
        self._filled = R.copy()
        
        # greedy forward search for final patterns
        l = [mdl(B, r) for r in R]
        idx = [np.argmin(l)]
        opt = l[idx[-1]]
        while True:
            l = [mdl(B, R[idx + [i,], :]) for i in range(R.shape[0])]
            if np.min(l) >= opt:
                break
            idx.append(np.argmin(l))
            opt = l[idx[-1]]
        R = R[idx, :]
        self._detected = R.copy()
    
        # 3. refine patterns
        refined = bishrinkgrow(B, R)

        # ensure edge cover is minimal
        iscover = []
        u = [set(np.flatnonzero(r)) for r in refined]
        for i, s in enumerate(u):
            cover = set().union(*(u[j] for j in range(len(u)) if j != i))
            if not s.issubset(cover):
                iscover.append(i)
        refined = refined[iscover, :]

        self._refined = refined
        self.opt_mdl = mdl(B, self._refined)

        self.communities = [np.flatnonzero(r).tolist() for r in self._refined]
        self.communities = [np.sort([order[i] for i in c]) for c in self.communities]


