import numpy as np
from numba import jit, float64
import numpy.linalg as la
from mvpose.compiler.numba_types import float64_2d_const


@jit([float64[:, :](float64[:, :], float64[:, :])], nopython=True, nogil=True)
def draw_vector_field(U, V):
    """
        draw the single part affinity field for a given limb
    :param U: numpy.array: hxw, X-direction
    :param V: numpy.array: hxw, Y-direction
    :return:
    """
    assert U.shape == V.shape
    h, w = U.shape
    Vec = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            vx = U[y, x]
            vy = V[y, x]
            N = la.norm(np.array([vx, vy]))
            Vec[y, x] = N

    return Vec


@jit([float64[:, :](float64_2d_const, float64_2d_const)], nopython=True, nogil=True)
def draw_vector_field_readonly(U, V):
    """
        draw the single part affinity field for a given limb
    :param U: numpy.array: hxw, X-direction
    :param V: numpy.array: hxw, Y-direction
    :return:
    """
    assert U.shape == V.shape
    h, w = U.shape
    Vec = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            vx = U[y, x]
            vy = V[y, x]
            N = la.norm(np.array([vx, vy]))
            Vec[y, x] = N

    return Vec