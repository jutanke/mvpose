import numpy as np
import numpy.linalg as la
from numba import jit, float64
from mvpose.compiler.numba_types import  int32_2d_const, float64_2d_const
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_MAP_IDX
from mvpose.pose_estimation.limb_weights import LimbWeights
from mvpose.plot.limbs import draw_vector_field
from mvpose.geometry import vector_fields as vec


def calculate_line_integral_elementwise(candA, candB, mapx, mapy, normalize=True):
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

    if normalize:
        mapx, mapy = vec.clamp_to_1(mapx, mapy)

    mapx = np.expand_dims(mapx, axis=2)
    mapy = np.expand_dims(mapy, axis=2)
    score_mid = np.concatenate([mapx, mapy], axis=2)
    h,w,_ = score_mid.shape

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
            x_ = int(round(x))
            if x_ == w:
                x_ = x_ -1
            y_ = int(round(y))
            if y_ == h:
                y_ = y_ -1
            Lc = score_mid[y_, x_]
            W[i] += Lc @ d

    return W/mid_num


def calculate_line_integral(candA, candB, mapx, mapy, normalize=True):
    """
        calculates the line integral for points
    :param candA: [ (x,y), (x,y), ...]
    :param candB: [ (x,y), (x,y), ...]
    :param mapx: w x h
    :param mapy: w x h
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

    if normalize:
        mapx, mapy = vec.clamp_to_1(mapx, mapy)

    mapx = np.expand_dims(mapx, axis=2)
    mapy = np.expand_dims(mapy, axis=2)
    score_mid = np.concatenate([mapx, mapy], axis=2)

    #W = np.zeros((nA * nB, 1))
    W = np.zeros((nA, nB))

    #cur_item = 0
    for i in range(nA):
        for j in range(nB):
            d = np.subtract(candB[j], candA[i])
            norm = la.norm(d)
            if norm == 0:
                #cur_item += 1
                continue
            d = np.divide(d, norm)

            iterX = np.linspace(candA[i][0], candB[j][0], mid_num)
            iterY = np.linspace(candA[i][1], candB[j][1], mid_num)

            for x, y in zip(iterX, iterY):
                x_ = int(round(x))
                y_ = int(round(y))
                Lc = score_mid[y_, x_]
                #W[cur_item] += Lc @ d
                W[i, j] += Lc @ d

            #cur_item += 1

    return W/mid_num


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

    total = 0
    for i in range(len(limbSeq)):
        a = limbSeq[i, 0]
        b = limbSeq[i, 1]
        nA = max(0, lookup[a, 1] - lookup[a, 0])
        nB = max(0, lookup[b, 1] - lookup[b, 0])
        total += nA * nB

    W = np.zeros((total, 1))

    cur_item = 0
    for k in range(len(limbSeq)):
        a = limbSeq[k, 0]
        b = limbSeq[k, 1]
        candA = data[lookup[a, 0]:lookup[a, 1]]
        candB = data[lookup[b, 0]:lookup[b, 1]]

        l = mapIdx[k, 0]
        r = mapIdx[k, 1] + 1
        score_mid = pafs[:, :, l:r]
        # score_mid = pafs[:,:,[x for x in mapIdx[k]]]

        nA = len(candA)
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
