import numpy as np


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