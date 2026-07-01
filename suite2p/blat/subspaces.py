import numpy as np
from scipy import linalg, stats, signal
import copy
from sklearn.cluster import DBSCAN, OPTICS

zero_one_norm = lambda x: (x - np.min(x, axis=1, keepdims=True)) / np.ptp(x, axis=1, keepdims=True)

def rand_remap(stack):
    """
    Apply random remapping (circular) on stack
    """
    shift = np.random.choice(stack.shape[1], size=stack.shape[0], replace=True)
    ret = copy.deepcopy(stack)
    for i, s in enumerate(shift):
        ret[i, :] = np.roll(ret[i, :], s)
    return ret

def angles(stack, k=2):
    """
    Computer matrix of subspace angles from stack
    """
    stack = copy.deepcopy(stack)
    stack = stats.zscore(stack, axis=1)
    # stack = zero_one_norm(stack)
    stack[np.isnan(stack)] = 0
    mat = np.zeros((stack.shape[1] - k + 1, stack.shape[1] - k + 1, k))
    for i in range(stack.shape[1] - k + 1):
        for j in range(stack.shape[1] - k + 1):
            s1 = stack[:, i:i+k]
            s2 = stack[:, j:j+k]

            b1 = linalg.orth(s1)
            b2 = linalg.orth(s2)

            mat[i, j, :] = linalg.subspace_angles(b1, b2)

    return mat

def delta_angles(s1, s2, k=2):
    """
    Return different in subspace angles in radians
    """
    assert np.all(s1.shape == s2.shape), 's1 and s2 must have same dimensions'
    theta1 = angles(s1, k=k)
    theta2 = angles(s2, k=k)
    idx = np.triu(np.ones_like(theta1[:, :, 0], dtype=bool), k)
    delta = theta1 - theta2
    ret = np.array([])
    for i in range(delta.shape[2]):
        ret = np.append(ret, (delta[:, :, i])[idx])
    return ret


def rayleigh(theta, correction=True):
    '''
    Rayleigh test for uniformity of distribution around circular.
    '''
    assert len(theta) >= 4, 'n >= 4 is needed to satisfy asymptotic result'

    n = len(theta)
    R = np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2) / n
    if correction:
        z = (1 - 1 / 2 / n) * 2 * n * R**2 + n * R**4 / 2
    else:
        z = 2 * n * R**2

    p = np.exp(-z / 2)
    return z, p

def bonferroni(p):
    p *= len(p)
    p = np.clip(p, 0, 1)
    return p

def floodfill(reachability, order, capillary = .0, min_size=10):
    '''
    Detect clusters in ordered OPTICS reachability vector using a custom
    flood fill algorithm.
    '''
    landscape = reachability[order]
    landscape[np.isinf(landscape)] = np.max(landscape[~np.isinf(landscape)])
    minima, _ = signal.find_peaks(-landscape)

    clusters = np.ones_like(landscape, dtype=int) * -1
    for i, m in enumerate(minima):
        ranger = [m-1, m+1]
        while 1:
            step = np.argmin([landscape[ranger[0]-1], landscape[ranger[1]+1]])
            if landscape[ranger[step]] - landscape[ranger[step] + ((step - .5) * 2).astype(int)] > capillary:
                break
            ranger[step] = ranger[step] + ((step - .5) * 2).astype(int)

        border = np.max(landscape[ranger[0]:ranger[1]+1])
        if landscape[ranger[1] + 1] == border:
            ranger[1] += 1
        while landscape[ranger[0] + 1] == border:
            ranger[0] += 1
        if ranger[1] - ranger[0] + 1 >= min_size:
            clusters[ranger[0]:ranger[1]+1] = i

    for new, old in enumerate(np.unique(clusters)[1:]):
        clusters[clusters == old] = new

    return clusters[np.argsort(order)]


def isoclinic_alignment(A, B, k=2, p_correct=False):
    assert np.all(A.shape == B.shape), 'A and B must have same shape'

    A = copy.deepcopy(A)
    B = copy.deepcopy(B)

    # normalize to unit vectors
    A /= linalg.norm(A, axis=0)
    B /= linalg.norm(B, axis=0)

    # build reference vector
    reference = np.append([1,], np.zeros((A.shape[0] - 1,)))
    reference = reference[:, np.newaxis]

    # isoclinic rotation alignment of A to reference (i.e., circ roll)
    mag = []
    for i in range(A.shape[0]):
        M = np.roll(np.eye(A.shape[0]), i, axis=0)
        mag.append(reference.T @ (M @ A))

    mag = np.array(mag).squeeze()
    shift = np.argmax(mag, axis=0)

    rot = []
    for i in shift:
        M = np.roll(np.eye(A.shape[0]), i, axis=0)
        rot.append(M)
    rot = np.array(rot)

    # rotate A and B to reference
    for i, r in enumerate(rot):
        A[:, i] = r @ A[:, i]
        B[:, i] = r @ B[:, i]

    # compute delta vectors and cluster them using OPTICS
    delta = A - B
    clusts = OPTICS(min_samples=10, metric='euclidean').fit(delta.T)
    clusters = floodfill(clusts.reachability_, clusts.ordering_, min_size=5)

    # get rid of spurious clusters by rayleigh test
    theta = shift / 80 * np.pi * 2
    pvals = [rayleigh(theta[clusters == i])[1] for i in np.unique(clusters)[1:]]
    pvals = np.array(pvals)
    if p_correct:
        pvals = bonferroni(pvals)
    for c in np.flatnonzero(pvals > .05):
        clusters[clusters == c] = -1
    for new, old in enumerate(np.unique(clusters)[1:]):
        clusters[clusters == old] = new

    # perform subspace alignment
    aligned = np.zeros_like(B)
    rot_mats = []
    for c in np.unique(clusters)[1:]:
        a = A[:, clusters == c]
        b = B[:, clusters == c]

        M = a @ b.T
        u, s, v = linalg.svd(M, full_matrices=True)
        # R = u @ np.diag(s) @ v
        D = np.diag(np.ones((M.shape[0],)))
        D[-1, -1] = np.round(linalg.det(u @ v))
        R = u @ D @ v # Kabsch algorithm

        aligned[:, clusters == c] = R @ b
        rot_mats.append(R)
        # M = a.T @ b
        # u, s, v = linalg.svd(M, full_matrices=True)
        # R = u @ v
        # aligned[:, clusters == c] = b @ R


    # roll back rotations
    for i, r in enumerate(rot):
        A[:, i] = r.T @ A[:, i]
        B[:, i] = r.T @ B[:, i]
        aligned[:, i] = r.T @ aligned[:, i]

    return aligned, clusters, delta, rot_mats