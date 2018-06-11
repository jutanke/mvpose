from mvpose.algorithm.peaks2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.algorithm.limbs3d import Limbs3d
from mvpose.algorithm.candidate_selection import CandidateSelector
from collections import namedtuple
from time import time
from scipy.spatial import KDTree
from numba import jit, vectorize, float64
from math import exp, sqrt, pi
import numpy as np
import numpy.linalg as la
import sklearn.metrics as mt
from scipy.optimize import linear_sum_assignment


@vectorize([float64(float64, float64, float64, float64)])
def gauss3d(x, y, z, sigma):
    N = 1/sqrt(2**3 * sigma**6 * pi**2)
    return N * exp(- (x*x + y*y + z*z) / sigma**2)


@jit([float64[:](float64[:], float64[:, :], float64)], nopython=True, nogil=True)
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

    result = np.zeros((3,))
    div = 0

    for i in range(num):
        result += Nx[i ,0:3] * G[i] * Nx[i, 3]
        div += G[i] * Nx[i, 3]

    return result /div


def estimate(Calib, heatmaps, pafs, settings,
             radius, sigma, max_iterations, between_distance,
             debug):
    """
        Brute-Force graph partitioning algorithm (np-hard)
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param radius: radius for meanshift density estimation
    :param sigma: width of the gaussian in the meanshift
    :param max_iterations: cut-of threshold for meanshift
    :param between_distance: maximal distance between two points of a cluster
    :param debug:
    :return:
    """
    # -------- step 1 --------
    # calculate 2d candidates
    # ------------------------
    _start = time()
    cand2d = Candidates2D(heatmaps, Calib,
                          threshold=settings.hm_detection_threshold)
    _end = time()
    if debug:
        print('step 1: elapsed', _end - _start)

    # -------- step 2 --------
    # triangulate 2d candidates
    # ------------------------
    _start = time()
    triangulation = Triangulation(cand2d, Calib, settings.max_epi_distance)
    _end = time()
    if debug:
        print('step 2: elapsed', _end - _start)

    # -------- step 3 --------
    # meanshift
    # ------------------------
    _start = time()
    eps = 0.1 / settings.scale_to_mm
    meanshift = Meanshift(triangulation.peaks3d_weighted,
                          float(radius), float(sigma), max_iterations, eps,
                          between_distance)
    _end = time()
    if debug:
        print('step 3: elapsed', _end - _start)

    # -------- step 4 --------
    # calculate 3d limb weights
    # ------------------------
    _start = time()
    limbs3d = Limbs3d(meanshift.centers3d,
                      Calib, pafs,
                      settings.limb_seq,
                      settings.sensible_limb_length,
                      settings.limb_map_idx,
                      oor_marker=-999999999)
    _end = time()
    if debug:
        print('step 4: elapsed', _end - _start)

    # -------- step 5 --------
    # linear sum assignment
    # ------------------------
    _start = time()
    modes3d = meanshift.centers3d
    pid = 0
    # represents all modes and their respective pid (-1 => no person)
    modes_to_person = [[-1] * len(x) for x in modes3d]

    for lid, (k1, k2) in enumerate(settings.limb_seq):
        W = -limbs3d[lid]  # weight for the modes
        rows, cols = linear_sum_assignment(W)
        for a, b in zip(rows, cols):

            if W[a, b] > 0:
                continue

            pid1 = modes_to_person[k1][a]
            pid2 = modes_to_person[k2][b]

            if pid1 == -1 and pid2 == -1:
                modes_to_person[k1][a] = pid
                modes_to_person[k2][b] = pid
                pid += 1
            elif pid1 == -1:
                modes_to_person[k1][a] = pid2
            elif pid2 == -1:
                modes_to_person[k2][b] = pid1
            else:  # merge?
                pass  # TODO: we need to do something here!

    humans = {}
    for jid, pids in enumerate(modes_to_person):
        for idx, pid in enumerate(pids):
            if pid not in humans:
                humans[pid] = [None] * cand2d.n_joints
            humans[pid][jid] = modes3d[jid][idx]

    human_candidates = []
    for v in humans.values():
        count_valid = 0
        for i in range(cand2d.n_joints):
            count_valid = count_valid if v[i] is None else count_valid + 1
            v[i] = v[i][0:3] if v[i] is not None else None

        if count_valid > settings.min_nbr_joints:
            human_candidates.append(v)

    _end = time()
    if debug:
        print('step 5: elapsed', _end - _start)

    # -------- step 6 --------
    # candidate selection  "filter out bad detections"
    # ------------------------
    _start = time()
    candSelector = CandidateSelector(
        human_candidates, heatmaps,
        Calib, settings.min_nbr_joints)
    _end = time()
    if debug:
        print('step 5: elapsed', _end - _start)

    # ------------------------
    # finalize
    # ------------------------
    if debug:
        Debug = namedtuple('Debug', [
            'candidates2d',
            'triangulation',
            'meanshift',
            'limbs3d',
            'human_candidates'
        ])
        Debug.candidates2d = cand2d
        Debug.triangulation = triangulation
        Debug.meanshift = meanshift
        Debug.limbs3d = limbs3d
        Debug.human_candidates = human_candidates
        return Debug, candSelector.persons
    else:
        return candSelector.persons


class Meanshift:
    """
        applies meanshift to 3d data
    """

    def __init__(self, peaks3d, radius, sigma, max_iterations, eps, between_distance):
        """
        :param peaks3d: [ [ (x,y,z,w1,w2), ...], ... ]
        :param radius: radius for meanshift density estimation
        :param sigma: width of the gaussian in the meanshift
        :param max_iterations: cut-of threshold for meanshift
        :param eps: iteration epsilon
        :param between_distance: maximal distance between two points of a cluster
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
            pts3d[:, 3] = peaks3d[jid][:, 3] * peaks3d[jid][:, 4]  # TODO try different functions

            all_centers = pts3d.copy()

            for i, p in enumerate(pts3d):
                y = p[0:3]

                # --- meanshift algorithm ---
                lookup = KDTree(pts3d[:, 0:3])

                y_t = y
                if max_iterations > 0:
                    for _ in range(max_iterations):
                        Nx = pts3d[lookup.query_ball_point(y_t, r=radius)]
                        y_tp1 = m(y_t, Nx, sigma)
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
                Peaks[i, 3] = total_w
            centers3d[jid] = Peaks

        for i in range(n_joints):
            if all_clusters[i] is None:
                all_clusters[i] = []

        self.centers3d = centers3d
        self.all_clusters = all_clusters
