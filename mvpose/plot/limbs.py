import numpy as np
from numba import jit, float64
import numpy.linalg as la
from mvpose.data.default_limbs import DEFAULT_LIMB_SEQ


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


def draw_mscoco_human(ax, human, cam, color, alpha=1):
    """
    :param ax
    :param human: [ (x,y), None, ... ]
    :param cam: {mvpose.geometry.camera}
    :param color
    :param alpha
    :return:
    """
    assert len(human) == 18
    for jid, pt3d in enumerate(human):
        if pt3d is None:
            continue
        pt = cam.projectPoints([pt3d])[0]
        marker = '*'
        if jid in [2, 3, 4, 8, 9, 10, 14, 16]:
            marker = '_'
        elif jid in [5, 6, 7, 11, 12, 13, 15, 17]:
            marker = '|'
        ax.scatter(pt[0], pt[1], color=color, marker=marker, alpha=alpha)

        for a, b in DEFAULT_LIMB_SEQ:
            ptA = human[a]
            ptB = human[b]
            if ptA is not None and ptB is not None:
                x_a, y_a = cam.projectPoints([ptA[0:3]])[0]
                x_b, y_b = cam.projectPoints([ptB[0:3]])[0]
                ax.plot([x_a, x_b], [y_a, y_b], color=color, alpha=alpha)


# @jit([float64[:, :](float64_2d_const, float64_2d_const)], nopython=True, nogil=True)
# def draw_vector_field_readonly(U, V):
#     """
#         draw the single part affinity field for a given limb
#     :param U: numpy.array: hxw, X-direction
#     :param V: numpy.array: hxw, Y-direction
#     :return:
#     """
#     assert U.shape == V.shape
#     h, w = U.shape
#     Vec = np.zeros((h, w))
#     for x in range(w):
#         for y in range(h):
#             vx = U[y, x]
#             vy = V[y, x]
#             N = la.norm(np.array([vx, vy]))
#             Vec[y, x] = N
#
#     return Vec