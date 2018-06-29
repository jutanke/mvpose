from ortools.linear_solver import pywraplp as mip
from reid import reid
import numpy.linalg as la
import numpy as np
from time import time
from mvpose.geometry import geometry as gm

# loading the model takes quite a while, thus
# we try to do it right at the start to buffer
# the model
reid_model = reid.ReId()


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
                 valid_person_bb_area):
        """
            Tracking using human poses
        :param Calibs:
        :param Ims:
        :param humans_candidates:
        :param valid_person_bb_area: valid area in [pixel] over
            which a person reprojection into an image is considered
            valid
        :param debug
        """
        n_frames = len(Ims)
        assert len(humans_candidates) == n_frames
        assert len(Calibs) == n_frames

        # =====================================
        # calculate Edge costs
        # =====================================
        _, _, H, W, _ = Ims.shape
        ImgsA = []
        ImgsB = []
        pairs = []  # t1, pid1, cid1, t2 pid2, cid2
        _start = time()
        for t1 in range(n_frames - 1):
            for t2 in range(t1 + 1, n_frames):
                for pidA, candA in enumerate(humans_candidates[t1]):
                    for pidB, candB in enumerate(humans_candidates[t2]):
                        for cidA, camA in enumerate(Calibs[t1]):
                            for cidB, camB in enumerate(Calibs[t2]):
                                aabb_A = get_bb(camA, candA, W, H)
                                aabb_B = get_bb(camB, candB, W, H)
                                if  gm.aabb_area(aabb_A) > valid_person_bb_area \
                                    and gm.aabb_area(aabb_B) > valid_person_bb_area:

                                    tx, ty, bx, by = aabb_A
                                    ImgsA.append(Ims[t1][cidA][ty: by, tx: by])
                                    tx, ty, bx, by = aabb_A
                                    ImgsB.append(Ims[t2][cidB][ty: by, tx: by])
                                    pairs.append(t1, pidA, cidA,
                                                 t2, pidB, cidB)
        _end = time()
        self.ImgsA = ImgsA
        self.ImgsB = ImgsB
        self.pairs = pairs
        if debug:
            print('\t[gp:step 1] elapsed', _end - _start)
            print('\t\t# boxes to compare:', len(ImgsA))

        # =====================================
        # build graph
        # =====================================
        solver = mip.Solver('t', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        Xi = {}

        for t1 in range(n_frames - 1):
            for t2 in range(t1 + 1, n_frames):
                pass
