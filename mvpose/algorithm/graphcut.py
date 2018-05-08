from mvpose.pose_estimation import limb_weights
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH, DEFAULT_MAP_IDX
from mvpose.algorithm.meanshift import find_all_modes
from mvpose.pose_estimation import part_affinity_fields as mvpafs
import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment
import mvpose.pose_estimation.heatmaps as mvhm
from mvpose.candidates import peaks as mvpeaks
from scipy.special import comb
import cv2
from time import time
from ortools.linear_solver import pywraplp


class GraphCutSolver:

    def __init__(self, Heatmaps, Pafs, Calib, r, sigma=-1,
                 limbSeq=DEFAULT_LIMB_SEQ,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
                 limbMapIdx=DEFAULT_MAP_IDX, debug=False
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
        CAMERA_NORM = comb(n_cameras, 2)  # this is needed to make the 3d pafs be between -1 .. 1
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
        _start = time()
        self.peaks2d = []
        self.peaks2d_undistorted = []

        self.undistort_maps = []
        self.Calib_undistorted = []

        n_joints = -1

        for cid, cam in enumerate(Calib):
            hm = Heatmaps[cid]
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
        _end = time()
        if debug:
            print("[GRAPHCUT] step1 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 2: triangulate all points
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _start = time()
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
        _end = time()
        if debug:
            print("[GRAPHCUT] step2 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 3: reproject all 3d points onto all 2d views
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # IMPORTANT: the points have to be distorted so that
        #            we can safely work with the pafs (which
        #            are only defined in the distorted world)

        # defined as follows:
        #       (x, y, score1, score2, point-line-dist1, point-line-dist1)

        _start = time()
        # TODO: this is bug-heavy memorywise...
        self.candidates2d_undistorted = [[np.zeros((0, 6))] * n_joints] * n_cameras
        self.candidates2d = [[np.zeros((0, 6))] * n_joints] * n_cameras

        for cid, cam in enumerate(self.Calib_undistorted):
            K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)
            assert distCoef == 0

            Cand2d = self.candidates2d_undistorted[cid].copy()
            Cand2d_dist = self.candidates2d[cid].copy()
            assert len(Cand2d) == n_joints
            assert len(Cand2d_dist) == n_joints

            mapx, mapy = self.undistort_maps[cid]

            for k in range(n_joints):
                Pts3d = Peaks3d[k][:, 0:3]
                pts2d, mask = gm.reproject_points_to_2d(Pts3d, rvec, tvec, K, w, h)

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
        _end = time()
        if debug:
            print("[GRAPHCUT] step3 elapsed:", _end - _start)


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 4: calculate the weights for the limbs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _start = time()
        self.limbs3d = [None] * n_limbs

        assert len(limbSeq) == len(sensible_limb_length)
        for idx,((a,b), (length_min, length_max), (pafA, pafB)) in \
                enumerate(zip(limbSeq, sensible_limb_length, limbMapIdx)):
            # 3d peaks are setup as follows:
            #       (x,y,z,score1,score2,p2l-dist1,p2l-dist2)
            candA3d = Peaks3d[a]
            candB3d = Peaks3d[b]

            nA = len(candA3d)
            nB = len(candB3d)

            W = np.zeros((nA, nB))

            for cid, cam in enumerate(Calib):
                K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)

                U = Pafs[cid,:,:,pafA]
                V = Pafs[cid,:,:,pafB]

                ptsA2d, maskA = gm.reproject_points_to_2d(
                    candA3d[:,0:3], rvec, tvec, K, w, h, distCoef=distCoef, binary_mask=True)
                ptsB2d, maskB = gm.reproject_points_to_2d(
                    candB3d[:,0:3], rvec, tvec, K, w, h, distCoef=distCoef, binary_mask=True)
                maskA = maskA == 1
                maskB = maskB == 1

                for i, (ptA, ptA3d, is_A_on_screen) in enumerate(zip(ptsA2d, candA3d, maskA)):
                    if not is_A_on_screen:
                        continue
                    ptA = np.expand_dims(ptA, axis=0)
                    for j, (ptB, ptB3d, is_B_on_screen) in enumerate(zip(ptsB2d, candB3d, maskB)):
                        if not is_B_on_screen:
                            continue
                        distance = la.norm(ptA3d[0:3] - ptB3d[0:3])
                        if length_min < distance < length_max:
                            ptB = np.expand_dims(ptB, axis=0)
                            line_int = mvpafs.calculate_line_integral(ptA, ptB, U, V)
                            score = np.squeeze(line_int) / CAMERA_NORM
                            W[i,j] += score

            self.limbs3d[idx] = W
        _end = time()
        if debug:
            print("[GRAPHCUT] step4 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 5: create optimization problem
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # creating poses is done in stages: first we optimize the
        # right limb candidates



