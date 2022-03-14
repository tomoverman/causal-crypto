import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as cor


def get_lagged_vectors(X, E, tau=1):
    """
    Given length L sequence X={X(0), ... X(L-1)},
    return an array of lagged vectors:
        <X(t), X(t-tau), X(t-(E+1)tau)>
    for t=(E-1)tau to t=L-1
    Input:
        X: length L array
        E: embedding dimension
        tau: step size
    Returns:
        ndarray of shape (L-(E-1)*tau, E)
        Each row is a lagged vector.
    """
    screen = np.flip(tau * np.arange(E)) + np.arange(len(X) - (E - 1) * tau)[:, None]
    return X[screen]


def get_nearest_neighbors(M, v, k):
    """
    Inputs:
        M: ndarray of shape (N, E), consisting of N vectors of dimension E.
        v: vector of length E.
        k: integer, number of nearest neighbors.
    Returns:
        idxs: indices of the nearest neighbors in the matrix M
        nns: ndarray of shape (k, E) consisting of the k nearest-neighbors of v
             in the set M.
        dists: the distances of the nearest k vectors, sorted in increasing order.
    """
    distances = np.linalg.norm(M - v, axis=1)
    idxs = np.argsort(distances)[:k]
    return idxs, M[idxs], distances[idxs]


class CCM:

    def __init__(self, X, Y, E, tau=1):
        self.X = X
        self.Y = Y
        self.E = E
        self.tau = tau
        self.Mx = get_lagged_vectors(X, E, tau)
        self.pred_domain = range(len(self.X) - len(self.Mx), len(self.X))
        self.nns_cache = {}  # maps index of lagged vectors to indices of E nearest neighbors

    def predict(self, t):
        """Predict the value of Y(t) using CCM algorithm"""

        # Get lagged vector corresponding to time t
        xt_idx = self.time_to_idx(t)
        if xt_idx < 0 or xt_idx >= len(self.Mx):
            # print(f"Cannot predict for t={t}")
            return None
        xt_lagged = self.Mx[xt_idx]

        # Compute nearest neighbors
        if xt_idx in self.nns_cache:
            nn_idxs, dists = self.nns_cache[xt_idx]
        else:
            nn_idxs, _, dists = get_nearest_neighbors(np.delete(self.Mx, xt_idx, 0), xt_lagged, self.E + 1)
            # add 1 to indices of vectors after x_idx
            self.nns_cache[xt_idx] = (nn_idxs + (nn_idxs >= xt_idx), dists)
            nn_idxs = (nn_idxs + (nn_idxs >= xt_idx))

        # Find the Y values corresponding to each lagged vector
        yts = self.Y[self.idx_to_time(nn_idxs)]

        return self.y_pred(dists, yts)

    def y_pred(self, dists, yts):
        if dists[0] == 0:
            dists[0] = 1e-12  # prevent divide by zero if d=0
        us = np.exp(-dists / dists[0])
        ws = us / np.sum(us)
        return np.dot(yts, ws)

    def time_to_idx(self, t):
        """Converts the time t to the corresponding index of the lagged vector in Mx."""
        return t - self.pred_domain[0]

    def idx_to_time(self, idx):
        """Converts the index of a lagged vector in Mx to its corresponding time."""
        return idx + self.pred_domain[0]


#
#
# # xs = np.linspace(0, 10, 1000)
# xs = np.linspace(0, 10, 100)
# xs = np.linspace(0, 100, 100)
# xs = np.linspace(0, 100, 1000)
#
# print(xs)
# X = np.sin(xs)
# Y = np.cos(xs)
# # X = np.array([0, 1, 2, 0,  1, 3,     0.1, 0.8, 1, 2.5, 12, 13, 20, 0, 1, 2])
# # Y = np.array([0, 0, 0, 10, 0, -0.04, 0.1, 9, 0, -0.5, 0.1, 13, 0, 0, 0.1, 14])
# E = 3
# tau = 1
#
#
# #test with stuff from fig 1 in the paper
# L = 3000
# X = np.zeros(L)
# Y = np.zeros(L)
# X[0] = 0.2
# Y[0] = 0.4
# for i in range(L-1):
# 	X_new = asymmetric(np.array([X[i], Y[i]]), 3.8, 3.5, -0.02, -0.1)
# 	X[i+1] = X_new[0]
# 	Y[i+1] = X_new[1]
#
#
# ccm = CCM(X, Y, E, tau)
#
# print(ccm.X)
# print(ccm.Y)
# print(ccm.Mx)
# print(ccm.predict(4))
# print(ccm.nns_cache)
#
#
# Y_pred = []
# for i in range(len(X)):
# 	y_pred = ccm.predict(i)
# 	Y_pred.append(y_pred)
# Y_pred = np.array(Y_pred)
#
# r, p = cor(Y[2:],Y_pred[2:])
# print(r) #should be about .3 for L=500, .8 for L=3000
# print(p)
#
# plt.plot(X, ':', label='X', marker='.')
# plt.plot(Y, label='Y', marker='.')
# plt.plot(Y_pred, label='$\hat{Y}$', marker='.')
# plt.plot()
# plt.legend()
# plt.show()





