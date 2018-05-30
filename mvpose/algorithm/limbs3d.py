import numpy as np
import mvpose.geometry.geometry as gm
from scipy.special import comb
import numpy.linalg as la
from numba import jit, float64


@jit([float64[:,:](float64[:,:,], float64[:,:,], float64[:,:,], float64[:,:,])], nopython=True)
def calculate_line_integral_elementwise(candA, candB, mapx, mapy):
    """
        calculates the line integral for points element-wise
    :param candA: [ (x,y), (x,y), ...]
    :param candB: [ (x,y), (x,y), ...]
    :param mapx: w x h
    :param mapy: w x h
    :param normalize:
    :return:
    """
    assert mapx.shape == mapy.shape
    assert len(candA.shape) == 2
    assert len(candB.shape) == 2
    assert candA.shape[1] == 2
    assert candB.shape[1] == 2
    assert len(mapx.shape) == 2
    mid_num = 10
    nA = len(candA)
    nB = len(candB)
    assert nA == nB

    h, w = mapx.shape
    score_mid = np.zeros((h,w,2))
    score_mid[:,:,0] = mapx
    score_mid[:,:,1] = mapy

    W = np.zeros((nA, 1))

    for i in range(nA):
        d = np.subtract(candB[i], candA[i])
        norm = la.norm(d)
        if norm == 0:
            continue
        d = np.divide(d, norm)

        iterX = np.linspace(candA[i][0], candB[i][0], mid_num)
        iterY = np.linspace(candA[i][1], candB[i][1], mid_num)

        for x, y in zip(iterX, iterY):
            x_ = min(int(round(x)), w-1)
            y_ = min(int(round(y)), h-1)
            Lc = score_mid[y_, x_]
            W[i] += Lc @ d

    return W/mid_num


class Limbs3d:
    """
    3d limb weight generation
    """

    def __init__(self, peaks3d, Calib, Pafs,
                 limbSeq, sensible_limb_length, limbMapIdx):
        """

        :param peaks3d: [ [ (x,y,z,w1,w2) ... ], ... ] * n_joints
        :param Calib: [ camera, ..]
        :param Pafs: Part affinity fields
        :param limbSeq:
        :param sensible_limb_length:
        :param limbMapIdx:
        """
        n_cameras, h, w, _ = Pafs.shape
        n_limbs = len(limbSeq)
        assert n_cameras == len(Calib)
        assert len(limbSeq) == len(sensible_limb_length)
        CAMERA_NORM = comb(n_cameras, 2)  # this is needed to make the 3d pafs be between -1 .. 1
        self.limbs3d = [None] * n_limbs

        for idx, ((a, b), (length_min, length_max), (pafA, pafB)) in \
                enumerate(zip(limbSeq, sensible_limb_length, limbMapIdx)):
            # 3d peaks are setup as follows:
            #       (x,y,z,score1,score2)
            candA3d = peaks3d[a]
            candB3d = peaks3d[b]

            nA = len(candA3d)
            nB = len(candB3d)

            W = np.zeros((nA, nB))

            if nA > 0 and nB > 0:
                for cid, cam in enumerate(Calib):
                    U = Pafs[cid, :, :, pafA]
                    V = Pafs[cid, :, :, pafB]
                    ptsA2d, maskA = cam.projectPoints(candA3d[:, 0:3], withmask=True)
                    ptsB2d, maskB = cam.projectPoints(candB3d[:, 0:3], withmask=True)
                    maskA = maskA == 1
                    maskB = maskB == 1

                    ptA_candidates = []
                    ptB_candidates = []
                    pair_candidates = []

                    for i, (ptA, ptA3d, is_A_on_screen) in enumerate(zip(ptsA2d, candA3d, maskA)):
                        if not is_A_on_screen:
                            continue
                        for j, (ptB, ptB3d, is_B_on_screen) in enumerate(zip(ptsB2d, candB3d, maskB)):
                            if not is_B_on_screen:
                                continue
                            distance = la.norm(ptA3d[0:3] - ptB3d[0:3])
                            if length_min < distance < length_max:
                                ptA_candidates.append(ptA)
                                ptB_candidates.append(ptB)
                                pair_candidates.append((i, j))

                    if len(ptA_candidates) > 0:
                        line_int = calculate_line_integral_elementwise(
                            np.array(ptA_candidates), np.array(ptB_candidates), U, V)
                        line_int = np.clip(line_int, a_min=-1, a_max=1)
                        assert len(line_int) == len(pair_candidates)
                        line_int = np.squeeze(line_int / CAMERA_NORM)
                        if len(line_int.shape) == 0:  # this happens when line_int.shape = (1, 1)
                            line_int = np.expand_dims(line_int, axis=0)
                        for score, (i, j) in zip(line_int, pair_candidates):
                            W[i, j] += score

            self.limbs3d[idx] = W

    def __getitem__(self, item):
        return self.limbs3d[item]