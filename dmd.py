# Code used from https://towardsdatascience.com/dynamic-mode-decomposition-for-multivariate-time-series-forecasting-415d30086b4b
import numpy as np
from pydmd import DMD


def DMD(data, r):
    ## Build data matrices
    X1 = data[:, : -1]
    X2 = data[:, 1:]
    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices=False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)

    return A_tilde, Phi, A, Q

def DMD_pred(data, r, pred_step):
    N, T = data.shape
    _, Phi, A, Q = DMD(data, r)
    mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
    for s in range(1,mat.shape[1]):
        mat[:, s] = (A @ mat[:, s - 1]).real

    return mat
