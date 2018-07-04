from ortools.linear_solver import pywraplp as mip
import numpy.linalg as la
import numpy as np
from time import time
from mvpose.geometry import geometry as gm


def distance3d_humans(human1, human2):
    """
        calculates the distance between two humans
    :param human1: [ (x, y, z, ?), ... ]
    :param human2:  -"-
    :return:
    """
    h1 = []
    h2 = []
    for j1, j2 in zip(human1, human2):
        if j1 is not None:
            h1.append(j1[0:3])
        if j2 is not None:
            h2.append(j2[0:3])
    p1 = np.mean(h1, axis=0)
    p2 = np.mean(h2, axis=0)
    return la.norm(p1 - p2)


def get_bb(cam, human, w, h):
    """
        gets the aabb from the human using the camera
    :param cam:
    :param human:
    :param w: image width
    :param h: image height
    :return:
    """
    points3d = []
    for joints in human:
        if joints is not None:
            points3d.append(joints)
    points2d = cam.projectPoints(np.array(points3d))
    points2d = np.round(points2d)
    max_x = np.clip(int(np.max(points2d[:, 0])), a_max=w-1, a_min=0)
    min_x = np.clip(int(np.min(points2d[:, 0])), a_max=w-1, a_min=0)
    max_y = np.clip(int(np.max(points2d[:, 1])), a_max=h-1, a_min=0)
    min_y = np.clip(int(np.min(points2d[:, 1])), a_max=h-1, a_min=0)
    return min_x, min_y, max_x, max_y


class GraphPartitioningTracker:

    def __init__(self, Calibs, Ims, humans_candidates, debug,
                 tracking_setting):
        """
            Tracking using human poses
        :param Calibs:
        :param Ims:
        :param humans_candidates:
        :param valid_person_bb_area: valid area in [pixel] over
            which a person reprojection into an image is considered
            valid
        :param tracking_setting: {mvpose.algorithm.settings.Tracking_Settings}
        :param debug
        """
        n_frames = len(Ims)
        assert len(humans_candidates) == n_frames
        assert len(Calibs) == n_frames
        low_spec_mode = tracking_setting.low_spec_mode

        # _start = time()
        # self.reid_model = reid.ReId()
        # _end = time()
        # if debug:
        #     print('\t[gp:step 1] elapsed', _end - _start)

        # =====================================
        # calculate Edge costs
        # =====================================
        valid_person_bb_area = tracking_setting.valid_person_bb_area
        max_moving_distance = tracking_setting.max_moving_distance_per_frame
        moving_factor_increase = tracking_setting.moving_factor_increase_per_frame
        _, _, H, W, _ = Ims.shape

        ImgsA = []
        ImgsB = []
        pairs = []  # t1, pid1, cid1, t2 pid2, cid2
        _start = time()
        for t1 in range(n_frames - 1):
            for t2 in range(t1 + 1, n_frames):
                dt = t2 - t1
                max_d = max_moving_distance * dt * moving_factor_increase
                for pidA, candA in enumerate(humans_candidates[t1]):
                    for pidB, candB in enumerate(humans_candidates[t2]):
                        # -- when the two person candidates are too far
                        # -- away from each other (depending on dt) we
                        # -- will not make them linkable candidates
                        distance = distance3d_humans(candA, candB)
                        if distance > max_d:
                            continue

                        for cidA, camA in enumerate(Calibs[t1]):
                            for cidB, camB in enumerate(Calibs[t2]):
                                aabb_A = get_bb(camA, candA, W, H)
                                aabb_B = get_bb(camB, candB, W, H)
                                if gm.aabb_area(aabb_A) < valid_person_bb_area \
                                        or gm.aabb_area(aabb_B) < valid_person_bb_area:
                                    # the bounding box must have a minimal area in the
                                    # camera view to be considered
                                    continue


                                tx, ty, bx, by = aabb_A
                                imga = Ims[t1][cidA][ty: by, tx: bx]
                                ImgsA.append(imga)
                                tx, ty, bx, by = aabb_B
                                imgb = Ims[t2][cidB][ty: by, tx: bx]
                                ImgsB.append(imgb)
                                pairs.append((t1, pidA, cidA,
                                             t2, pidB, cidB))
        _end = time()

        self.ImgsA = ImgsA
        self.ImgsB = ImgsB
        self.pairs = pairs
        if debug:
            print('\t[gp:step 1] elapsed', _end - _start)
            print('\t\t# boxes to compare:', len(ImgsA))

        # =====================================
        # predict similarity
        # =====================================
        _start = time()
        if low_spec_mode:
            scores = []
            for A, B in zip(ImgsA, ImgsB):
                score = tracking_setting.reid_model.predict(A, B)
                scores.append(score)
        else:
            scores = tracking_setting.reid_model.predict(ImgsA, ImgsB)
        _end = time()
        self.scores = scores
        assert len(scores) == len(ImgsA)

        if debug:
            print('\t[gp:step 2] elapsed', _end - _start)

        # =====================================
        # build graph
        # =====================================
        # solver = mip.Solver('t', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        # Xi = {}
        #
        # for t1 in range(n_frames - 1):
        #     for t2 in range(t1 + 1, n_frames):
        #         pass
