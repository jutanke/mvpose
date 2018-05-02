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
        n_cams, _, _, n_limbs = Pafs.shape
        n_limbs = n_limbs/2
        assert r > 0
        assert n_limbs == len(DEFAULT_LIMB_SEQ)
        assert n_cams == len(Calib)
        assert n_cams == len(Heatmaps)
        assert n_cams >= 3, 'The algorithm expects at least 3 views'
        if sigma == -1:
            r = sigma

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 1: get all peaks
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.peaks2d = []
        self.peaks2d_undistorted = []

        self.undistort_maps = []
        self.Calib_undistorted = []

        for cid, cam in enumerate(Calib):
            hm = Heatmaps[cid]
            paf = Pafs[cid]
            peaks = mvhm.get_all_peaks(hm)
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 2: triangulate all points
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

