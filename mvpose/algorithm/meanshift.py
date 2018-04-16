import numpy as np
import numpy.linalg as la
import mvpose.math as mvmath
from scipy.spatial import KDTree
from numba import jit, float64


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
