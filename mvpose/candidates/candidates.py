import numpy as np
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo


class Candidates3d:
    """
        Represents the 3d candidates
    """

    def __init__(self):
        self.peaks3d = None
        self.joint_counters = None
        self.triangulation_lookup = None

    def get_3d_points(self, k):
        """
            returns the 3d points for the given joint k
        :param k: joint index
        :return:
        """
        return self.peaks3d[k]

    def triangulate(self, peaks, limbs, Calib):
        """
            triangulate all points in the cameras
        :param peaks: [{Peaks}]
        :param limbs: [{LimbWeights}]
        :param Calib: [{Camera}] The camera parameters
                    MUST BE undistorted!
        :return:
        """
        assert len(peaks) == len(limbs)
        assert len(peaks) == len(Calib)

        n_cameras = len(Calib)

        for cam1 in range(n_cameras-1):
            K1, rvec1, tvec1, distCoef1 = \
                gm.get_camera_parameters(Calib[cam1])
            assert distCoef1 == 0
            peaks1 = peaks[cam1]

            for cam2 in range(cam1+1, n_cameras):
                K2, rvec2, tvec2, distCoef2 = \
                    gm.get_camera_parameters(Calib[cam2])
                assert distCoef2 == 0
                peaks2 = peaks[cam2]

                current_3d_peaks = stereo.triangulate(
                    peaks1, K1, rvec1, tvec1, peaks2, K2, rvec2, tvec2)

                if self.peaks3d is None:
                    self.peaks3d = current_3d_peaks
                else:
                    self.peaks3d.merge(current_3d_peaks)
