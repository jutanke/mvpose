from mvpose.pose_estimation import limb_weights
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH
from mvpose.algorithm.meanshift import find_all_modes
import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment
import mvpose.pose_estimation.heatmaps as mvhm
from mvpose.candidates import peaks as mvpeaks
import cv2


class GraphCutSolver:

    def __init__(self, Heatmaps, Pafs, Calib, r, sigma=-1,
                 limbSeq=DEFAULT_LIMB_SEQ,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
                 ):
        """
            Extract 3d pose from images and cameras
        :param Heatmaps: list of heatmaps
        :param Pafs: list of part affinity fields
        :param Calib: list of calibrations per camera
        :param r:
        :param sigma:
        :param limbSeq:
        :param sensible_limb_length:
        """
        n_cameras, _, _, n_limbs = Pafs.shape
        n_limbs = int(n_limbs/2)
        assert r > 0
        assert n_limbs == len(DEFAULT_LIMB_SEQ)
        assert n_cameras == len(Calib)
        assert n_cameras == len(Heatmaps)
        assert n_cameras >= 3, 'The algorithm expects at least 3 views'
        if sigma == -1:
            r = sigma

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 1: get all peaks
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.peaks2d = []
        self.peaks2d_undistorted = []

        self.undistort_maps = []
        self.Calib_undistorted = []

        n_joints = -1

        for cid, cam in enumerate(Calib):
            hm = Heatmaps[cid]
            paf = Pafs[cid]
            peaks = mvhm.get_all_peaks(hm)
            if n_joints < 0:
                n_joints = peaks.n_joints
            else:
                assert n_joints == peaks.n_joints
            self.peaks2d.append(peaks)

            # -- undistort peaks --
            K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)
            hm_ud, K_new = gm.remove_distortion(hm, cam)
            h,w,_ = hm.shape

            mapx, mapy = \
                cv2.initUndistortRectifyMap(
                K, distCoef, None, K_new, (w, h), 5)
            self.undistort_maps.append((mapx, mapy))

            peaks_undist = mvpeaks.Peaks.undistort(peaks, mapx, mapy)
            self.peaks2d_undistorted.append(peaks_undist)

            self.Calib_undistorted.append({
                'K': K_new,
                'distCoeff': 0,
                'rvec': rvec,
                'tvec': tvec
            })

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 2: triangulate all points
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Peaks3d = [np.zeros((0, 7))] * n_joints

        for cam1 in range(n_cameras - 1):
            K1, rvec1, tvec1, distCoef1 = \
                gm.get_camera_parameters(self.Calib_undistorted[cam1])
            assert distCoef1 == 0
            peaks1 = self.peaks2d_undistorted[cam1]

            for cam2 in range(cam1 + 1, n_cameras):
                K2, rvec2, tvec2, distCoef2 = \
                    gm.get_camera_parameters(self.Calib_undistorted[cam2])
                assert distCoef2 == 0
                peaks2 = self.peaks2d_undistorted[cam2]

                peaks3d = stereo.triangulate_with_weights(
                    peaks1, K1, rvec1, tvec2,
                    peaks2, K2, rvec2, tvec2
                )
                assert len(peaks3d) == n_joints

                for k in range(n_joints):
                    Peaks3d[k] = np.concatenate(
                        [Peaks3d[k], peaks3d[k]], axis=0
                    )

        self.peaks3d_weighted = Peaks3d