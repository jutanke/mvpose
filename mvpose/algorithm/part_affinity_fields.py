import numpy as np
from numba import jit
import math
import numpy.linalg as la


@jit('double[:,:](double[:,:], double[:,:], double[:,:,:])', nopython=True, nogil=True)
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
