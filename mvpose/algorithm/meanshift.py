import numpy as np
import numpy.linalg as la
import mvpose.math as mvmath
from scipy.spatial import KDTree
from numba import jit, float64
import mvpose.geometry.clustering as cl


def find_all_modes(X, r, sigma=None,
                   max_iterations=1000, lim=1, between_distance=50):
    """
        finds all modes in the data
    :param X: [ (x,y,z,score), ... ]
    :param r:
    :param sigma:
    :param max_iterations:
    :param lim: how many views must see a point to be valid
    :param between_distance: the maximal distance between
            two points of a cluster
    :return:
    """
    if X is None or X.shape[0] == 0:  # no detections at all!
        return np.array([]), []

    r = float(r)
    if sigma is None:
        sigma = r

    # calculate the peaks from ALL points
    # TODO check if this can be done probabilistically,
    # TODO e.g. using the weights of the points as Pr()
    n = len(X)
    all_centers = np.zeros((n,4))
    for i, p in enumerate(X):
        y = p[0:3]
        w = p[3]
        cx,cy,cz = meanshift(y, X,
                             r=r,
                             sigma=sigma,
                             max_iterations=max_iterations)
        all_centers[i] = [cx,cy,cz,w]

    # --- merge centers ---
    Clusters = cl.cluster(all_centers[:,0:3], between_distance=between_distance)
    Peaks = []
    for cluster in Clusters:
        if len(cluster) > lim:
            Peaks.append(cluster)

    Modes = []
    for peaks in Peaks:
        N = all_centers[peaks]
        total_w = np.sum(N[:,3])
        Pts = N[:, 0:3]
        W = np.expand_dims(N[:, 3]/total_w, axis=1)
        Modes.append(np.sum(Pts * W, axis=0))

    return np.array(Modes), Peaks


def meanshift(y, X, r, sigma=None, max_iterations=1000):
    """

    :param y:
    :param X: all data [ (x,y,z,score)...]
    :param r: radius
    :param sigma:
    :param max_iterations:
    :return:
    """
    eps = 0.000001
    if sigma is None:
        sigma = r

    lookup = KDTree(X[:, 0:3])

    y_t = y
    if max_iterations > 0:
        for i in range(max_iterations):
            Nx = X[lookup.query_ball_point(y_t, r=r)]
            y_tp1 = m(y_t, Nx, sigma)
            step_size = la.norm(y_tp1 - y_t)
            if step_size < eps:
                break
            y_t = y_tp1
        return y_tp1
    else:  # run until 'convergence'
        while True:
            Nx = X[lookup.query_ball_point(y_t, r=r)]
            y_tp1 = m(y_t, Nx, sigma)
            step_size = la.norm(y_tp1 - y_t)
            if step_size < eps:
                return y_tp1
            y_t = y_tp1


@jit([float64[:](float64[:], float64[: ,:], float64)], nopython=True, nogil=True)
def m(y, Nx, sigma):
    num = len(Nx)
    G = mvmath.gauss3d(y[0]-Nx[: ,0], y[1]-Nx[: ,1], y[2]-Nx[: ,2], sigma)

    result = np.zeros((3,))
    div = 0

    for i in range(num):
        result += Nx[i ,0:3] * G[i] * Nx[i, 3]
        div += G[i] * Nx[i, 3]

    return result /div
