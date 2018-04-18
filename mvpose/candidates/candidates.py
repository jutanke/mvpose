from mvpose.pose_estimation import limb_weights
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ


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

    def triangulate(self, peaks, limbs, Calib, limbSeq=DEFAULT_LIMB_SEQ):
        """
            triangulate all points in the cameras
        :param peaks: [{Peaks}]
        :param limbs: [{LimbWeights}]
        :param Calib: [{Camera}] The camera parameters MUST BE undistorted!
        :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
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
        self.lw = limb_weights.LimbWeights3d(self.peaks3d, IDX_PAIRS, LIMB_PAIRS, limbSeq)
        return self.peaks3d, self.lw