from scipy.spatial import KDTree
from numba import jit, vectorize, float32
from math import exp, sqrt, pi
import numpy as np
import numpy.linalg as la
import sklearn.metrics as mt


@vectorize([float32(float32, float32, float32, float32)])
def gauss3d(x, y, z, sigma):
    N = 1/sqrt(2**3 * sigma**6 * pi**2)
    return N * exp(- (x*x + y*y + z*z) / sigma**2)


@jit([float32[:](float32[:], float32[:, :], float32)], nopython=True, nogil=True)
def m(y, Nx, sigma):
    """
        applies a meanshift step
    :param y: (x,y,z)
    :param Nx: [ (x,y,z,w), ... ]
    :param sigma: gaussian width
    :return:
    """
    num = len(Nx)
    G = gauss3d(y[0]-Nx[:, 0], y[1]-Nx[:, 1], y[2]-Nx[:, 2], sigma)

    result = np.zeros((3,), np.float32)
    div = 0

    for i in range(num):
        result += Nx[i ,0:3] * G[i] * Nx[i, 3]
        div += G[i] * Nx[i, 3]

    div = np.array(div, np.float32)
    result = result / div
    return result


class Meanshift:
    """
        applies meanshift to 3d data
    """

    def __init__(self, peaks3d, radius, sigma, max_iterations, eps, between_distance, n_cameras):
        """
        :param peaks3d: [ [ (x,y,z,w1,w2), ...], ... ]
        :param radius: radius for meanshift density estimation
        :param sigma: width of the gaussian in the meanshift
        :param max_iterations: cut-of threshold for meanshift
        :param eps: iteration epsilon
        :param between_distance: maximal distance between two points of a cluster
        :param n_cameras: number of cameras
        """
        n_joints = len(peaks3d)
        LARGE = 999999999

        all_clusters = [None] * n_joints
        centers3d = [np.zeros((0, 4))] * n_joints
        for jid in range(n_joints):
            # -- apply meanshift for each joint type --
            n = len(peaks3d[jid])
            if n == 0:
                continue

            pts3d = np.zeros((n, 4))
            pts3d[:, 0:3] = peaks3d[jid][:, 0:3]
            pts3d[:, 3] = peaks3d[jid][:, 3] * peaks3d[jid][:, 4] * 2 # TODO try different functions

            all_centers = pts3d.copy()

            for i, p in enumerate(pts3d):
                y = p[0:3]

                # --- meanshift algorithm ---
                lookup = KDTree(pts3d[:, 0:3])

                y_t = y
                if max_iterations > 0:
                    for _ in range(max_iterations):
                        Nx = pts3d[lookup.query_ball_point(y_t, r=radius)]
                        y_tp1 = m(y_t.astype('float32'),
                                  Nx.astype('float32'), sigma)
                        step_size = la.norm(y_t - y_tp1)
                        if step_size < eps:
                            break
                        y_t = y_tp1
                else:  # run until 'convergence'
                    while True:
                        Nx = pts3d[lookup.query_ball_point(y_t, r=radius)]
                        y_tp1 = m(y_t, Nx, sigma)
                        step_size = la.norm(y_t - y_tp1)
                        if step_size < eps:
                            break
                        y_t = y_tp1
                # ---------------------------
                all_centers[i, 0:3] = y_t

            # --- now cluster the data ---
            Clusters = []
            n = len(all_centers)

            # largest distance value, is used to ensure that points are not
            # clustered with themselves
            Padding = np.diag([LARGE] * n)
            # create the n x n pairwise distance matrix
            D = mt.pairwise_distances(all_centers[:, 0:3], all_centers[:, 0:3])
            D += Padding  # see explanation above

            left, right = np.where(D < between_distance)

            ca = [-1] * n  # cluster assignment

            cur_cluster_id = 0

            for a, b in zip(left, right):
                if ca[a] == -1 and ca[b] == -1:
                    ca[a] = cur_cluster_id
                    ca[b] = cur_cluster_id
                    cur_cluster_id += 1
                elif ca[a] == -1:
                    ca[a] = ca[b]
                elif ca[b] == -1:
                    ca[b] = ca[a]
                else:  # merge clusters!
                    for idx in range(n):
                        if ca[idx] == ca[b]:
                            ca[idx] = ca[a]

            lookup = dict()  # check at what index a cluster is set
            for i in range(n):
                cid = ca[i]
                if cid in lookup:
                    Clusters[lookup[cid]].append(i)
                else:
                    Clusters.append([i])
                    if cid >= 0:
                        lookup[cid] = len(Clusters) - 1
            all_clusters[jid] = Clusters
            # ----------------------------

            Peaks = np.zeros((len(Clusters), 4))
            for i, cluster in enumerate(Clusters):
                N = all_centers[cluster]
                total_w = np.sum(N[:, 3])
                Pts = N[:, 0:3]
                W = np.expand_dims(N[:, 3] / total_w, axis=1)
                Peaks[i, 0:3] = np.sum(Pts * W, axis=0)
                Peaks[i, 3] = total_w / n_cameras
            centers3d[jid] = Peaks

        for i in range(n_joints):
            if all_clusters[i] is None:
                all_clusters[i] = []

        self.centers3d = centers3d
        self.all_clusters = all_clusters
