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


def draw_mscoco_human3d(ax, human, color, alpha=1):
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
        pt = pt3d[0:3]
        marker = '*'
        if jid in [2, 3, 4, 8, 9, 10, 14, 16]:
            marker = '_'
        elif jid in [5, 6, 7, 11, 12, 13, 15, 17]:
            marker = '|'
        ax.scatter(pt[0], pt[1], pt[2], color=color, marker=marker, alpha=alpha)

        for a, b in DEFAULT_LIMB_SEQ:
            ptA = human[a]
            ptB = human[b]
            if ptA is not None and ptB is not None:
                x_a, y_a, z_a = ptA[0:3]
                x_b, y_b, z_b = ptB[0:3]
                ax.plot([x_a, x_b], [y_a, y_b], [z_a, z_b],
                        color=color, alpha=alpha)


def draw_mscoco_human2d(ax, human, color, alpha=1, linewidth=1):
    """ draws a 2d person
    :param ax:
    :param human:
    :param color:
    :param alpha:
    :param linewidth:
    :return:
    """
    RIGHT = {2, 3, 4, 8, 9, 10, 14, 16}
    LEFT = {5, 6, 7, 11, 12, 13, 15, 17}
    lcolor = lighten_color(color)
    rcolor = color
    assert len(human) == 18
    for jid, pt2d in enumerate(human):
        if pt2d is None:
            if jid in RIGHT:
                clr = rcolor
            elif jid in LEFT:
                clr = lcolor

            ax.scatter(*pt2d, color=clr, alpha=alpha)

    for a, b in DEFAULT_LIMB_SEQ:
        ptA = human[a]
        ptB = human[b]
        if ptA is not None and ptB is not None:
            clr = rcolor
            if a in LEFT or b in LEFT:
                clr = lcolor
            x_a, y_a = ptA
            x_b, y_b = ptB

            if x_a == -1 or x_b == -1:
                continue

            ax.plot([x_a, x_b], [y_a, y_b], color=clr,
                    alpha=alpha, linewidth=linewidth)




def draw_mscoco_human(ax, human, cam, color, alpha=1, linewidth=1):
    """
    :param ax
    :param human: [ (x,y), None, ... ]
    :param cam: {mvpose.geometry.camera}
    :param color
    :param alpha
    :param linewidth
    :return:
    """
    assert len(human) == 18
    for jid, pt3d in enumerate(human):
        if pt3d is None:
            continue
        pt = cam.projectPoints(np.array([pt3d[0:3]]))[0]
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
                x_a, y_a = cam.projectPoints(np.array([ptA[0:3]]))[0]
                x_b, y_b = cam.projectPoints(np.array([ptB[0:3]]))[0]
                ax.plot([x_a, x_b], [y_a, y_b], color=color, alpha=alpha, linewidth=linewidth)


# Taken from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.65):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
