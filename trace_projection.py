from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds
import numpy as np
'''
objective: argmin_A = 0.5 \|A - Z\| + rho/2 \|Z\|* 
'''

def _my_svd(M, k, method):
    if method == 'randomized':
        U, S, V = randomized_svd(
            M, n_components=min(k, M.shape[1]-1), n_oversamples=20)
    elif method == 'arpack':
        U, S, V = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
    else:
        raise ValueError("wrong algorithm")
    return U, S, V



def trace_proj(X,k,alpha,mask):
    U, S, V = svds(X, k=min(k, min(X.shape) - 1))
    threshold_value = np.diag(S) - alpha/2
    diag_S = np.maximum(threshold_value,0)
    X_hat = np.linalg.multi_dot([U, diag_S, V])
    X_tn = np.linalg.matrix_rank(X_hat)
    # print(np.sqrt(np.sum(mask*(X - X_hat)**2)/np.sum(mask)))
    return X_hat,X_tn


def trace_projection(A,mask,alpha):
    Y = np.zeros_like(A)
    tau = 5 * np.sum(A.shape) / 2
    delta = 1.2 * np.prod(A.shape) / np.sum(mask)
    for _ in range(550):
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S = np.maximum(S - alpha/2, 0)
        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y += delta * mask * (A - X)
        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        print(np.sqrt(np.sum(mask * (X - A)**2)/np.sum(mask)))
        if recon_error < 0.005:
            break
    return X,np.linalg.matrix_rank(X)