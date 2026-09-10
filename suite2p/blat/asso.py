import numpy as np


def mdl_2factor(B, X=None, Y=None):
    '''
    Minimum description length for general two-factor matrix factorization (B approx X @ Y).
    '''
    def entropy(p):
        p = np.clip(p, 1e-12, 1. - 1e-12)
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    B = np.asarray(B, dtype=bool)
    m, n = B.shape

    if X is not None and Y is not None:
        X = np.asarray(X, dtype=bool)
        Y = np.asarray(Y, dtype=bool)

        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[np.newaxis, :]

        k = X.shape[1]
        p_x = np.mean(X, axis=0)
        p_y = np.mean(Y, axis=1)

        # Model description length for factors X (m x k) and Y (k x n)
        L = k * np.log2(m + 1) + m * np.sum(entropy(p_x))
        L += k * np.log2(n + 1) + n * np.sum(entropy(p_y))

        # Error description length over full m x n matrix
        N = m * n
        err = B ^ ((X @ Y) > 0)
        p = np.sum(err) / N
        L += np.log2(N + 1) + N * entropy(p)
    else:
        N = m * n
        p = np.sum(B) / N
        L = np.log2(N + 1) + N * entropy(p)

    return L


def asso_2factor(B, tau):
    '''
    ASSO algorithm for general two-factor matrix decomposition (B approx X @ Y).
    Returns factor matrices X (m x k), Y (k x n), and the MDL cost.
    '''
    B = np.asarray(B, dtype=bool)
    m, n = B.shape
    R = B.copy()

    # Pairwise column association matrix A (n x n)
    A = [[np.sum(a & b) / np.sum(a | b) if np.sum(a | b) > 0 else 0 for b in B.T] for a in B.T]
    A = np.array(A) > tau

    X_list = []
    Y_list = []
    opt = mdl_2factor(B)

    max_k = min(m, n)
    for _ in range(max_k):
        if len(A) == 0:
            break

        x_cands = []
        gains = []

        # For each candidate row vector y in A, compute optimal column vector x
        for y in A:
            row_gains = np.sum(R & y, axis=1) - np.sum((~R) & y, axis=1)
            x_cand = row_gains > 0
            gain = np.sum(row_gains[x_cand])

            x_cands.append(x_cand)
            gains.append(gain)

        idx = np.argmax(gains)
        best_y = A[idx, :]
        best_x = x_cands[idx]

        X_list.append(best_x)
        Y_list.append(best_y)

        A = np.delete(A, idx, axis=0)
        R = R ^ (R & np.outer(best_x, best_y))

        X_mat = np.column_stack(X_list)
        Y_mat = np.array(Y_list)

        L = mdl_2factor(B, X_mat, Y_mat)
        if L > opt:
            X_list.pop(-1)
            Y_list.pop(-1)
            break
        opt = L

    if len(X_list) == 0:
        return np.zeros((m, 1), dtype=bool), np.zeros((1, n), dtype=bool), opt

    X = np.column_stack(X_list)
    Y = np.array(Y_list)
    return X, Y, opt


def mdlasso_2factor(B, nsplits=20):
    '''
    MDL-ASSO model order selection for general two-factor matrix decomposition.
    Returns optimal factor matrices X (m x k) and Y (k x n).
    '''
    best_tau, min_l = 0.0, float('inf')
    for tau in np.linspace(0, 1, nsplits + 2)[1:-1]:
        _, _, l = asso_2factor(B, tau)
        if l < min_l:
            min_l, best_tau = l, tau

    X, Y, _ = asso_2factor(B, best_tau)
    return X, Y


def mdl(B, pat=None, X=None, Y=None):
    '''
    Minimum description length wrapper supporting both symmetric (pat) and two-factor (X, Y) cases.
    '''
    if X is not None or Y is not None:
        return mdl_2factor(B, X, Y)

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
        if B.ndim == 2 and B.shape[0] != B.shape[1]:
            return mdl_2factor(B)
        n = B.shape[0]
        N = n * (n + 1) / 2
        p = np.sum(np.triu(B)) / N
        L += np.log2(N + 1) + N * entropy(p)
    return L


def asso(B, tau, two_factor=False):
    '''
    ASSO algorithm. Automatically dispatches to two-factor decomposition if B is rectangular
    or if two_factor=True.
    '''
    if two_factor or (B.ndim == 2 and B.shape[0] != B.shape[1]):
        return asso_2factor(B, tau)

    R = B.copy()
    A = [[np.sum(a & b) / np.sum(a | b) if np.sum(a | b) > 0 else 0 for b in B] for a in B]
    A = np.array(A) > tau

    pat = []
    opt = mdl(B)
    for _ in range(B.shape[0]):
        gain = [np.sum(np.triu(R & np.outer(r, r))) - np.sum(np.triu(~R & np.outer(r, r))) for r in A]
        idx = np.argmax(gain)
        pat.append(A[idx, :])
        A = np.delete(A, idx, axis=0)
        R = R ^ (R & np.outer(pat[-1], pat[-1]))
        L = mdl(B, np.array(pat))
        if L > opt:
            pat.pop(-1)
            break
        opt = L

    if len(pat) == 0:
        return np.zeros((1, B.shape[0]), dtype=bool), opt

    return np.array(pat), opt


def mdlasso(B, nsplits=20, two_factor=False):
    '''
    MDL-ASSO wrapper. Automatically dispatches to two-factor decomposition if B is rectangular
    or if two_factor=True.
    '''
    if two_factor or (B.ndim == 2 and B.shape[0] != B.shape[1]):
        return mdlasso_2factor(B, nsplits)

    best_tau, min_l = 0.0, float('inf')
    for tau in np.linspace(0, 1, nsplits + 2)[1:-1]:
        _, l = asso(B, tau)
        if l < min_l:
            min_l, best_tau = l, tau

    return asso(B, best_tau)[0]
