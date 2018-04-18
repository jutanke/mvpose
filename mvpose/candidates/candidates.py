from mvpose.pose_estimation import limb_weights
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH
from mvpose.algorithm.meanshift import find_all_modes
import numpy as np


class Candidates3d:
    """
        Represents the 3d candidates
    """

    def __init__(self):
        self.peaks3d = None

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