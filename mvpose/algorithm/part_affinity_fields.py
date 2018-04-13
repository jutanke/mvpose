import numpy as np
import numpy.linalg as la
from numba import jit, float64
from mvpose.compiler.numba_types import  int32_2d_const, float64_2d_const
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_MAP_IDX


class LimbWeights:
    """
        represents the weight between joints (limbs)
    """

    def __init__(self, W, lookup, limbSeq):
        """
        :param W: calculated by {calculate_weights_for_all}
        :param lookup: extracted from {mvpose.algorithm.peaks}
        :param limbSeq: pre-defined: see {mvpose.algorithm.candidate_generator}
        """
        self.cost_by_limb = []
        total = 0
        for i in range(len(limbSeq)):
            a = limbSeq[i, 0];
            b = limbSeq[i, 1]
            nA = lookup[a, 1] - lookup[a, 0]
            nB = lookup[b, 1] - lookup[b, 0]
            total += nA * nB

        assert total == W.shape[0]  # sanity check

        cur_item = 0
        for i in range(len(limbSeq)):
            a = limbSeq[i, 0];
            b = limbSeq[i, 1]
            nA = lookup[a, 1] - lookup[a, 0]
            nB = lookup[b, 1] - lookup[b, 0]
            length = nA * nB

            assert cur_item + length <= total
            data = W[cur_item:cur_item + length].reshape((nA, nB))
            data.setflags(write=False)
            self.cost_by_limb.append(data)
            cur_item += length

        assert len(self.cost_by_limb) == len(limbSeq)

    def __getitem__(self, lid):
        return self.cost_by_limb[lid]


def calculate_limb_weights(peaks, pafs, limbSeq=DEFAULT_LIMB_SEQ, mapIdx=DEFAULT_MAP_IDX):
    """
        Calculates the weights of the limbs given the peaks

    :param peaks: all peaks for all joints for the given image
    :param pafs: complete part affinity field for the given image
    :param limbSeq: defines the joint pairs for a single limb
    :param mapIdx: maps the limb to the vector field
    :return:
    """
    lookup = peaks.lookup
    data = peaks.data

    W = calculate_weights_for_all(data, lookup, limbSeq, mapIdx, pafs)

    limbWeights = LimbWeights(W, lookup, limbSeq)
    return limbWeights


@jit([float64[:, :](float64_2d_const, int32_2d_const, int32_2d_const, int32_2d_const, float64[:, :, :])],
     nopython=True, nogil=True)
def calculate_weights_for_all(data, lookup, limbSeq, mapIdx, pafs):
    """

    :param data: extracted from {mvpose.algorithm.peaks}
    :param lookup: extracted from {mvpose.algorithm.peaks}
    :param limbSeq:
    :param mapIdx:
    :param pafs: part affinity field
    :return:
    """
    mid_num = 10
    n_limbs = limbSeq.shape[0]

    total = 0
    for i in range(len(limbSeq)):
        a = limbSeq[i, 0];
        b = limbSeq[i, 1]
        nA = lookup[a, 1] - lookup[a, 0]
        nB = lookup[b, 1] - lookup[b, 0]
        total += nA * nB

    W = np.zeros((total, 1))

    cur_item = 0
    for k in range(len(limbSeq)):
        a = limbSeq[k, 0];
        b = limbSeq[k, 1]
        candA = data[lookup[a, 0]:lookup[a, 1]]
        candB = data[lookup[b, 0]:lookup[b, 1]]

        l = mapIdx[k, 0];
        r = mapIdx[k, 1] + 1
        score_mid = pafs[:, :, l:r]
        # score_mid = pafs[:,:,[x for x in mapIdx[k]]]

        nA = len(candA);
        nB = len(candB)

        for i in range(nA):
            for j in range(nB):
                d = np.subtract(candB[j][:2], candA[i][:2])
                norm = la.norm(d)
                if norm == 0:
                    cur_item += 1
                    continue
                d = np.divide(d, norm)

                iterX = np.linspace(candA[i][0], candB[j][0], mid_num)
                iterY = np.linspace(candA[i][1], candB[j][1], mid_num)

                for x, y in zip(iterX, iterY):
                    x_ = int(round(x))
                    y_ = int(round(y))
                    Lc = score_mid[y_, x_]
                    W[cur_item] += Lc @ d

                cur_item += 1

    return W


@jit([float64[:,:](float64[:,:], float64[:,:], float64[:,:,:])], nopython=True, nogil=True)
def calculate_weights(candA, candB, score_mid):
    """
    calculates the weight for a given pair of candidates and a given part affinity field
    :param candA: {np.array([n x 2])} x/y position of first joint of the limb
    :param candB: {np.array([m x 2])} x/y position of second joint of the limb
    :param score_mid: {np.array([h x w x 2]} part affinity field for the given limb
    :return:
    """
    mid_num = 10
    nA = len(candA); nB = len(candB)
    W = np.zeros((nA, nB))
    for i in range(nA):
        for j in range(nB):
            d = np.subtract(candB[j][:2], candA[i][:2])
            norm = la.norm(d)
            if norm == 0:
                continue
            d = np.divide(d, norm)

            iterX = np.linspace(candA[i][0], candB[j][0], mid_num)
            iterY = np.linspace(candA[i][1], candB[j][1], mid_num)

            for x, y in zip(iterX, iterY):
                x_ = int(round(x))
                y_ = int(round(y))
                Lc = score_mid[y_ ,x_]
                W[i, j] += Lc@d
    return W
