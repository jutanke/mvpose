from mvpose.pose_estimation import limb_weights
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH
from mvpose.algorithm.meanshift import find_all_modes
import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment
import deprecation


class Candidates:

    def __init__(self, peaks, limbs, Calib, r, sigma=None,
                 limbSeq=DEFAULT_LIMB_SEQ,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
                 mode_between_distance=50,
                 threshold_drop_person=8,
                 threshold_nbr_multiview_modes=3):
        """
            triangulate all points in the cameras
        :param peaks: [{Peaks}]
        :param limbs: [{LimbWeights}]
        :param Calib: [{Camera}] The camera parameters MUST BE undistorted!
        :param r: {float} radius for meanshift
        :param sigma: {float} sigma for meanshift
        :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
        :param sensible_limb_length: {np.array[m x 2]} (low, high) of sensible limb length'
        :param mode_between_distance: the maximal distance between two points of a cluster
                        This is being used for clustering the detected mode of all points
        :param threshold_drop_person: {integer} if less then this threshold
                        items are found: drop the whole detection
        :param threshold_nbr_multiview_modes: {integer} define how many modes per human must be seen
                        from more than one camera pair
        :return: {Peaks3}, {LimbWeights3d}
        """
        n_cameras = len(Calib)
        n_joints = peaks[0].n_joints
        n_limbs = len(limbSeq)
        assert n_cameras == len(limbs)
        assert n_cameras == len(peaks)
        assert limbSeq.shape[1] == 2
        assert n_limbs == len(sensible_limb_length)

        POINTS_3d = [np.zeros((0, 4))] * n_joints  # per joint
        META = [np.zeros((0, 4), 'int32')] * n_joints  # per joint: [(cam1, cam2, idx1, idx2) ... ]
        # the meta data maps the points to their respective 2d camera pair

        self.peaks2d = peaks
        self.limbs2d = limbs

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # step 1: triangulate points
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for cam1 in range(n_cameras - 1):
            K1, rvec1, tvec1, distCoef1 = \
                gm.get_camera_parameters(Calib[cam1])
            assert distCoef1 == 0
            peaks1 = peaks[cam1]

            for cam2 in range(cam1 + 1, n_cameras):
                K2, rvec2, tvec2, distCoef2 = \
                    gm.get_camera_parameters(Calib[cam2])
                assert distCoef2 == 0
                peaks2 = peaks[cam2]

                current_3d_peaks, idx_pairs = stereo.triangulate_argmax(
                    peaks1, K1, rvec1, tvec1, peaks2, K2, rvec2, tvec2)
                assert len(current_3d_peaks) == n_joints

                for k in range(n_joints):
                    n = idx_pairs[k].shape[0]
                    pts3d = current_3d_peaks[k]
                    assert len(pts3d) == n

                    POINTS_3d[k] = np.concatenate(
                        [POINTS_3d[k], pts3d])

                    for i in range(n):
                        idx1 = idx_pairs[k][i, 0]
                        idx2 = idx_pairs[k][i, 1]
                        meta = np.expand_dims([cam1, idx1, cam2, idx2], axis=0)
                        META[k] = np.append(META[k], meta, axis=0)

        self.points3d = POINTS_3d

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # step 2: calculate weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        limbs3d = []
        for lid, (k1, k2) in enumerate(limbSeq):
            d_low, d_high = sensible_limb_length[lid]
            meta1 = META[k1]
            meta2 = META[k2]
            pts1 = POINTS_3d[k1]
            pts2 = POINTS_3d[k2]

            n = len(meta1)
            m = len(meta2)
            assert n == len(pts1)
            assert m == len(pts2)

            W = np.zeros((n, m))

            for u, (cam_a1, a1, cam_b1, b1) in enumerate(meta1):
                for v, (cam_a2, a2, cam_b2, b2) in enumerate(meta2):
                    assert cam_a1 != cam_b1 and cam_a2 != cam_b2
                    if cam_a1 == cam_a2 and cam_b1 == cam_b2:
                        p1_3d = pts1[u][0:3]
                        p2_3d = pts2[v][0:3]
                        W1 = limbs[cam_a1][lid]
                        W2 = limbs[cam_b1][lid]
                        distance = la.norm(p1_3d - p2_3d)

                        if d_low < distance < d_high:
                            W[u, v] = W1[a1, a2] + W2[b1, b2]
                        else:
                            W[u, v] = -9999999999

            limbs3d.append(W)
        self.limbs3d = limbs3d

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # step 3: find modes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r = float(r)
        if sigma is None:
            sigma = r

        modes3d = []

        for k in range(n_joints):
            sum
            modes = []

            Modes, Lookup = find_all_modes(POINTS_3d[k], r, sigma,
                                           lim=0, between_distance=mode_between_distance)
            n = len(Modes)
            for i in range(n):
                w = np.sum(POINTS_3d[k][Lookup[i]][:, 3])  # TODO: try out different techniques, e.g. mean
                item = [*Modes[i], w]
                modes.append(item)

            modes = np.array(modes)
            modes3d.append(
                (modes, Lookup)
            )

        self.modes3d = modes3d

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # step 4: calculate weight between modes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # W_limbs = [np.zeros((0, 0))] * n_limbs
        W_limbs = []
        for lid, (k1, k2) in enumerate(limbSeq):
            W_all_points = limbs3d[lid]  # Weight from all points

            d_low, d_high = sensible_limb_length[lid]

            # modes: [.. (x,y,z,score) ...]
            # lookup: [ [cluster1], [cluster2], ...]
            modes1, lookup1 = modes3d[k1]
            modes2, lookup2 = modes3d[k2]

            n = len(modes1)
            m = len(modes2)

            W = np.zeros((n, m))
            W_limbs.append(W)

            for u, (p1, idxs1) in enumerate(zip(modes1, lookup1)):
                for v, (p2, idxs2) in enumerate(zip(modes2, lookup2)):

                    distance = la.norm(p1[0:3] - p2[0:3])
                    if d_low < distance < d_high:
                        for i in idxs1:
                            for j in idxs2:
                                W[u, v] += W_all_points[i, j] * p1[3] * p2[3]
                    else:
                        W[u, v] = -99999999999

        assert len(W_limbs) == n_limbs
        self.mode_limbs = W_limbs

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # step 5: extract human pose
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        pid = 0
        # represents all modes and their respetive pid (-1 => no person)
        modes_to_person = [[-1] * len(x[0]) for x in modes3d]

        for lid, (k1, k2) in enumerate(limbSeq):
            W = -W_limbs[lid]  # weight for the modes

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
                    pass  # TODO: we need to do smthg here!

        # remove detections with too few points..
        count_pids = [0] * pid

        for pids in modes_to_person:
            for pid in pids:
                if pid >= 0:
                    count_pids[pid] += 1

        humans = {}
        for pid, count in enumerate(count_pids):
            if count > threshold_drop_person:
                # humans[pid] = [None] * len(modes_to_person)
                humans[pid] = {
                    'joints': [None] * len(modes_to_person),
                    'view_count': [0] * len(modes_to_person)
                }

        for k, pids in enumerate(modes_to_person):
            for idx, pid in enumerate(pids):
                if pid in humans:
                    humans[pid]['joints'][k] = modes3d[k][0][idx]
                    humans[pid]['view_count'][k] = len(modes3d[k][1][idx])

        # AT LEAST one point per human must be visible in more than 1 view pair,
        # otherwise it is probably just noise!
        rm = []
        for pid, human in humans.items():
            cnt = human['view_count']
            total = 0
            for c in cnt:
                if c > 1:
                    total += 1

            #threshold_nbr_multiview_modes

            #if np.max(cnt) <= 1:
            if total <= threshold_nbr_multiview_modes:
                rm.append(pid)

        for pid in rm:
            del humans[pid]

        self.humans = humans


@deprecation.deprecated(deprecated_in="0.1.0", removed_in="1.0.0")
class Candidates3d:
    """
        Represents the 3d candidates
    """

    def __init__(self):
        self.peaks3d = None
        self.lw = None

    def get_3d_points(self, k):
        """
            returns the 3d points for the given joint k
        :param k: joint index
        :return:
        """
        return self.peaks3d[k]

    def calculate_modes(self, r, sigma=None, limbSeq=DEFAULT_LIMB_SEQ):
        """
        :param r: radius
        :param sigma:
        :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
        :return:
        """
        n_limbs = len(limbSeq)
        assert self.lw is not None
        assert self.peaks3d is not None
        if sigma is None:
            sigma = r

        lw = self.lw
        pts3d = self.peaks3d

        # store [ .. ( [modes], [[idxs],...])...
        modes = []

        for k in range(pts3d.n_joints):
            modes.append(
                find_all_modes(
                    pts3d[k], r, sigma
                )
            )

        # [ [nxm] ... ]
        W_limbs = [None] * n_limbs

        for lid, (k1, k2) in enumerate(limbSeq):
            W = lw[lid]

            modes1 = modes[k1]
            modes2 = modes[k2]

            n = len(modes1[1]); m = len(modes2[1])
            W_modes = np.zeros((n,m))

            for a in range(n):
                items1 = modes1[1][a]
                for b in range(m):
                    items2 = modes2[1][b]
                    for i in items1:
                        for j in items2:
                            W_modes[a,b] += W[i,j]

            W_limbs[lid] = W_modes

        modes_only = []
        for modes, idxs in modes:
            modes_only.append(modes)

        return modes_only, W_limbs

    def triangulate(self, peaks, limbs, Calib,
                    limbSeq=DEFAULT_LIMB_SEQ,
                    sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH):
        """
            triangulate all points in the cameras
        :param peaks: [{Peaks}]
        :param limbs: [{LimbWeights}]
        :param Calib: [{Camera}] The camera parameters MUST BE undistorted!
        :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
        :param sensible_limb_length: {np.array[m x 2]} (low, high) of sensible limb length'
        :return: {Peaks3}, {LimbWeights3d}
        """
        n_cameras = len(Calib)
        assert n_cameras == len(limbs)
        assert n_cameras == len(peaks)
        assert limbSeq.shape[1] == 2

        IDX_PAIRS = []
        LIMB_PAIRS = []

        for cam1 in range(n_cameras - 1):
            K1, rvec1, tvec1, distCoef1 = \
                gm.get_camera_parameters(Calib[cam1])
            assert distCoef1 == 0
            peaks1 = peaks[cam1]
            limbs1 = limbs[cam1]

            for cam2 in range(cam1 + 1, n_cameras):
                K2, rvec2, tvec2, distCoef2 = \
                    gm.get_camera_parameters(Calib[cam2])
                assert distCoef2 == 0
                peaks2 = peaks[cam2]
                limbs2 = limbs[cam2]

                current_3d_peaks, idx_pairs = stereo.triangulate(
                    peaks1, K1, rvec1, tvec1, peaks2, K2, rvec2, tvec2)

                if self.peaks3d is None:
                    self.peaks3d = current_3d_peaks
                else:
                    self.peaks3d.merge(current_3d_peaks)

                LIMB_PAIRS.append((limbs1, limbs2))
                IDX_PAIRS.append(idx_pairs)

        # ---
        self.lw = limb_weights.LimbWeights3d(self.peaks3d, IDX_PAIRS, LIMB_PAIRS,
                                             limbSeq, sensible_limb_length)

        # sanity check
        for lid, (k1, k2) in enumerate(limbSeq):
            n,m = self.lw[lid].shape
            n_j1 = len(self.peaks3d[k1])
            n_j2 = len(self.peaks3d[k2])
            assert n == n_j1 and m == n_j2

        return self.peaks3d, self.lw