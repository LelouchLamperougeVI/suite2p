import numpy as np
from dataclasses import dataclass, field

def lk_tsp(dist, max_cand=5):
    n = dist.shape[0]
    cand = [
        sorted(range(n), key=lambda j: dist[i][j])[1 : max_cand + 1]
        for i in range(n)
    ]

    tour, unvisited = [0], set(range(1, n))
    while unvisited:
        tour.append(min(unvisited, key=lambda x: dist[tour[-1]][x]))
        unvisited.remove(tour[-1])

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
    Compute Jaccard distance matrix across rows of B.
    '''
    B_int = B.astype(np.int32)
    intersection = B_int @ B_int.T
    degrees = B_int.sum(axis=1)
    union = degrees[:, None] + degrees[None, :] - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        J = np.true_divide(intersection, union)
        J[union == 0] = 0.0  # Handle 0/0 division for isolated nodes
        
    return 1.0 - J


def nn2opt(D: np.ndarray, s: int = None, opt=True) -> list[int]:
    '''
    Nearest neighbour heuristic solution with 2-opt
    optimization to the travelling salesman problem.
    '''
    assert np.allclose(D, D.T), 'D must be symmetric'
    
    if D.shape[0] <= 1:
        return list(range(D.shape[0]))
    if s is None:
        s = int(np.argmax(np.sum(D, axis=0)))
    
    D_ = D.copy()
    D_[np.diag(np.ones((D_.shape[0],), dtype=bool))] = np.inf
    order = [s,]
    for _ in range(D_.shape[0] - 1):
        order.append(np.argmin(D_[order[-1], :]))
        D_[order[-2], :] = np.inf
        D_[:, order[-2]] = np.inf

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


def sort(B: np.ndarray) -> tuple[list[int], list[int]]:
    '''
    Sort asymmetric binary matrix to closely satisfy C1P
    along both row and column dimensions.
    '''
    row_order = nn2opt(jaccard(B))
    col_order = nn2opt(jaccard(B.T))
    return row_order, col_order


def mdl(B, pat_u=None, pat_v=None):
    '''
    Minimum Description Length for two-factor decomposition (U o V).
    '''
    def entropy(p):
        p = np.clip(p, 1e-12, 1. - 1e-12)
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    m, n = B.shape
    N = m * n

    if pat_u is not None and pat_v is not None:
        if len(pat_u.shape) == 1:
            pat_u = pat_u[np.newaxis, :]
        if len(pat_v.shape) == 1:
            pat_v = pat_v[np.newaxis, :]
            
        k_u, m_len = pat_u.shape
        k_v, n_len = pat_v.shape
        assert k_u == k_v, "Number of pattern factors must match"

        pu = np.mean(pat_u, axis=1)
        pv = np.mean(pat_v, axis=1)
        
        Lu = k_u * np.log2(m_len + 1) + m_len * np.sum(entropy(pu))
        Lv = k_v * np.log2(n_len + 1) + n_len * np.sum(entropy(pv))

        rec = (pat_u.T @ pat_v) > 0
        err = B ^ rec
        p = np.sum(err) / N
        L = Lu + Lv + np.log2(N + 1) + N * entropy(p)
    else:
        p = np.sum(B) / N
        L = np.log2(N + 1) + N * entropy(p)
        
    return L


def bishrinkgrow(B, pat_u, pat_v):
    '''
    Optimize factors (U, V) by minimizing MDL with greedy
    bidirectional shrink/growth in 2D space.
    '''
    def step_u(i, ind, direction, val, opt):
        ind += direction
        while 0 <= ind < pat_u.shape[1]:
            pat_u[i, ind] = val
            if (step_opt := mdl(B, pat_u, pat_v)) > opt:
                pat_u[i, ind] = not val
                break
            opt = step_opt
            ind += direction
        return opt, ind

    def step_v(i, ind, direction, val, opt):
        ind += direction
        while 0 <= ind < pat_v.shape[1]:
            pat_v[i, ind] = val
            if (step_opt := mdl(B, pat_u, pat_v)) > opt:
                pat_v[i, ind] = not val
                break
            opt = step_opt
            ind += direction
        return opt, ind

    for i in range(pat_u.shape[0]):
        # Optimize row pattern pat_u[i]
        if (seg_u := np.flatnonzero(pat_u[i])).size:
            cuts_u = np.flatnonzero(np.diff(seg_u) > 1) + 1
            for s in np.split(seg_u, cuts_u):
                left, right = s[0], s[-1]
                opt = mdl(B, pat_u, pat_v)
                opt, left = step_u(i, left, -1, True, opt)
                opt, _ = step_u(i, left, 1, False, opt)
                opt, right = step_u(i, right, 1, True, opt)
                opt, _ = step_u(i, right, -1, False, opt)

        # Optimize column pattern pat_v[i]
        if (seg_v := np.flatnonzero(pat_v[i])).size:
            cuts_v = np.flatnonzero(np.diff(seg_v) > 1) + 1
            for s in np.split(seg_v, cuts_v):
                left, right = s[0], s[-1]
                opt = mdl(B, pat_u, pat_v)
                opt, left = step_v(i, left, -1, True, opt)
                opt, _ = step_v(i, left, 1, False, opt)
                opt, right = step_v(i, right, 1, True, opt)
                opt, _ = step_v(i, right, -1, False, opt)

    return pat_u, pat_v


def quasi_cliques(B, pat_u, pat_v, noise):
    '''
    Extract quasi-bicliques from pattern pairs (pat_u, pat_v).
    Greedily eliminate the least connected rows/columns
    until noise floor is reached.
    '''
    if np.isnan(noise):
        noise = 0
    R_u = np.zeros_like(pat_u, dtype=bool)
    R_v = np.zeros_like(pat_v, dtype=bool)

    for i in range(pat_u.shape[0]):
        idx_u = np.flatnonzero(pat_u[i])
        idx_v = np.flatnonzero(pat_v[i])
        while len(idx_u) > 0 and len(idx_v) > 0:
            residual = B[np.ix_(idx_u, idx_v)]
            if np.mean(~residual) <= noise:
                break
            
            row_deg = np.mean(residual, axis=1)
            col_deg = np.mean(residual, axis=0)
            
            if np.min(row_deg) <= np.min(col_deg):
                idx_u = np.delete(idx_u, np.argmin(row_deg))
            else:
                idx_v = np.delete(idx_v, np.argmin(col_deg))

        if len(idx_u) > 0 and len(idx_v) > 0:
            R_u[i, idx_u] = True
            R_v[i, idx_v] = True

    return R_u, R_v


@dataclass
class BMF:
    B: np.ndarray
    communities: list[tuple[list[int], list[int]]] = field(default_factory=list)
    row_order: list[int] | np.ndarray = field(default_factory=list)
    col_order: list[int] | np.ndarray = field(default_factory=list)
    row_inverse_order: list[int] | np.ndarray = field(default_factory=list)
    col_inverse_order: list[int] | np.ndarray = field(default_factory=list)
    _quasi_u: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _quasi_v: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _detected_u: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _detected_v: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _refined_u: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _refined_v: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _filled_u: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    _filled_v: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    opt_mdl: float = 0.

    def __post_init__(self):
        B = self.B.copy()
        if B.dtype != bool:
            raise TypeError("B must be a boolean matrix")

        # 1. Sort matrix rows and columns independently by solving TSP
        row_order, col_order = sort(B)
        B = B[np.ix_(row_order, col_order)]
        self.B = B
        self.row_order = row_order
        self.col_order = col_order
        self.row_inverse_order = np.argsort(row_order)
        self.col_inverse_order = np.argsort(col_order)

        m, n = B.shape
        
        # Candidate pattern pair generation from rows and columns
        pat_u_list, pat_v_list = [], []
        for i in range(m):
            v_i = B[i, :].copy()
            if np.any(v_i):
                u_i = (B @ v_i) > 0
                pat_u_list.append(u_i)
                pat_v_list.append(v_i)
        for j in range(n):
            u_j = B[:, j].copy()
            if np.any(u_j):
                v_j = (B.T @ u_j) > 0
                pat_u_list.append(u_j)
                pat_v_list.append(v_j)

        if not pat_u_list:
            pat_u = np.zeros((0, m), dtype=bool)
            pat_v = np.zeros((0, n), dtype=bool)
        else:
            pat_u = np.array(pat_u_list, dtype=bool)
            pat_v = np.array(pat_v_list, dtype=bool)

        # 2. Minimize MDL and filter patterns
        spacing_u = np.linspace(0, m - 1, max(1, int(np.round(np.mean(B) * m))))
        spacing_v = np.linspace(0, n - 1, max(1, int(np.round(np.mean(B) * n))))
        uni_u = np.zeros((1, m), dtype=bool)
        uni_v = np.zeros((1, n), dtype=bool)
        uni_u[0, np.round(spacing_u).astype(int)] = True
        uni_v[0, np.round(spacing_v).astype(int)] = True
        thres = mdl(B, uni_u, uni_v)

        if pat_u.shape[0] > 0:
            pats = np.array([mdl(B, pat_u[[i]], pat_v[[i]]) for i in range(pat_u.shape[0])]) < thres
        else:
            pats = np.array([], dtype=bool)

        if np.sum(pats) == 0 or np.all(pats):
            noise = np.mean(B)
        else:
            unselected_u = pat_u[~pats]
            unselected_v = pat_v[~pats]
            noise = np.mean(np.mean(unselected_u, axis=1) * np.mean(unselected_v, axis=1))
            pat_u = pat_u[pats]
            pat_v = pat_v[pats]

        # Extract quasi-bicliques
        pat_u, pat_v = quasi_cliques(B, pat_u, pat_v, noise)
        self._quasi_u = pat_u.copy()
        self._quasi_v = pat_v.copy()

        fill_gaps = lambda x: np.array([np.convolve(r, np.ones(3), mode='same') >= 2 for r in x]) if len(x) > 0 else x
        pat_u = fill_gaps(pat_u)
        pat_v = fill_gaps(pat_v)
        self._filled_u = pat_u.copy()
        self._filled_v = pat_v.copy()

        # Greedy forward search for pattern subset
        if pat_u.shape[0] > 0:
            l = [mdl(B, pat_u[[i]], pat_v[[i]]) for i in range(pat_u.shape[0])]
            idx = [np.argmin(l)]
            opt = l[idx[-1]]
            while True:
                l = [mdl(B, pat_u[idx + [i]], pat_v[idx + [i]]) for i in range(pat_u.shape[0])]
                if np.min(l) >= opt:
                    break
                idx.append(np.argmin(l))
                opt = l[idx[-1]]
            pat_u = pat_u[idx]
            pat_v = pat_v[idx]

        self._detected_u = pat_u.copy()
        self._detected_v = pat_v.copy()

        # 3. Refine patterns
        refined_u, refined_v = bishrinkgrow(B, pat_u, pat_v)

        # Ensure minimal edge cover across cell products
        if refined_u.shape[0] > 0:
            iscover = []
            u_cells = [set(zip(*np.where(refined_u[i:i+1].T @ refined_v[i:i+1]))) for i in range(refined_u.shape[0])]
            for i, s in enumerate(u_cells):
                cover = set().union(*(u_cells[j] for j in range(len(u_cells)) if j != i))
                if not s.issubset(cover):
                    iscover.append(i)
            refined_u = refined_u[iscover]
            refined_v = refined_v[iscover]

        self._refined_u = refined_u
        self._refined_v = refined_v
        self.opt_mdl = mdl(B, self._refined_u, self._refined_v)

        self.communities = [
            (
                np.sort([row_order[r] for r in np.flatnonzero(ru)]).tolist(),
                np.sort([col_order[c] for c in np.flatnonzero(rv)]).tolist()
            )
            for ru, rv in zip(self._refined_u, self._refined_v)
        ]
