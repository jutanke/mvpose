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
        n_cameras, h, w, n_limbs = Pafs.shape
        n_limbs = int(n_limbs/2)
        assert r > 0
        assert n_limbs == len(DEFAULT_LIMB_SEQ)
        assert n_cameras == len(Calib)
        assert n_cameras == len(Heatmaps)
        assert h == Heatmaps.shape[1]
        assert w == Heatmaps.shape[2]
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 3: reproject all 3d points onto all 2d views
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # IMPORTANT: the points have to be distorted so that
        #            we can safely work with the pafs (which
        #            are only defined in the distorted world)

        # defined as follows:
        #       (x, y, score1, score2, point-line-dist1, point-line-dist1)

        # TODO: this is bug-heavy memorywise...
        self.candidates2d_undistorted = [[np.zeros((0, 6))] * n_joints] * n_cameras
        self.candidates2d = [[np.zeros((0, 6))] * n_joints] * n_cameras

        for cid, cam in enumerate(self.Calib_undistorted):
            K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)
            assert distCoef == 0
            distCoef = np.zeros((5,1))  # to match OpenCV API

            Cand2d = self.candidates2d_undistorted[cid].copy()
            Cand2d_dist = self.candidates2d[cid].copy()
            assert len(Cand2d) == n_joints
            assert len(Cand2d_dist) == n_joints

            mapx, mapy = self.undistort_maps[cid]

            for k in range(n_joints):
                Pts3d = Peaks3d[k][:, 0:3]
                Pts3d = Pts3d.astype('float64')

                pts2d, _ = cv2.projectPoints(Pts3d, rvec, tvec, K, distCoef)
                pts2d = np.squeeze(pts2d)

                # remove all points that are not visible in the view
                x = pts2d[:, 0]
                y = pts2d[:, 1]

                mask = (x > 0) * 1
                mask *= (x < w) * 1
                mask *= (y > 0) * 1
                mask *= (y < h) * 1
                mask = np.nonzero(mask)

                W = np.squeeze(Peaks3d[k][mask,3:].copy())
                pts2d = pts2d[mask]
                qq = np.concatenate([pts2d, W], axis=1)
                Cand2d[k] = qq
                assert Cand2d[k].shape[1] == 6

                pts2d_distorted = Cand2d[k].copy()
                dist_xy = gm.distort_points(pts2d[:,0:2], mapx, mapy)

                pts2d_distorted[:,0] = dist_xy[:,0]
                pts2d_distorted[:,1] = dist_xy[:,1]

                Cand2d_dist[k] = pts2d_distorted

            self.candidates2d_undistorted[cid] = Cand2d
            self.candidates2d[cid] = Cand2d_dist




