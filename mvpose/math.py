import numpy as np
from math import exp, sqrt, pi
from numba import jit, vectorize, float64


@jit([float64[:,:](float64[:], float64[:])], nopython=True, nogil=True)
def sum_1v1(a, b):
    """
        calculates the sum of all elements of one list against the elements
        of another one:

        c[i,j] = a[i] + b[j]
    :param a: [ ... ] \in |R
    :param b: [ ... ] \in |R
    :return: c
    """
    nA = len(a)
    nB = len(b)
    c = np.zeros((nA, nB))
    for x in range(nA):
        for y in range(nB):
            c[x,y] = a[x] + b[y]
    return c


@vectorize([float64(float64,float64,float64,float64)])
def gauss3d(x,y,z,sigma):
    N = 1/sqrt(2**3 * sigma**6 * pi**2)
    return N * exp(- (x*x + y*y + z*z) / sigma**2)


def cross_product_matrix3d(a):
    """ skew-symmetric matrix for a 3d vector

    :param a: a 3d vector
    :return:
    """
    M = np.zeros((3,3))
    M[0,1] = -a[2]
    M[1,0] = a[2]
    M[2,0] = -a[1]
    M[0,2] = a[1]
    M[2,1] = a[0]
    M[1,2] = -a[0]
    return M