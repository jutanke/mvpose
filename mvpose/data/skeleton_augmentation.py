"""
    Body structure for skeleton augmentation
    # * neck            (0)
    # * left-shoulder   (1)
    # * left-elbow      (2)
    # * left-hand       (3)
    # * right-shoulder  (4)
    # * right-elbow     (5)
    # * right-hand      (6)
    # * left-hip        (7)
    # * left-knee       (8)
    # * left-foot       (9)
    # * right-hip       (10)
    # * right-knee      (11)
    # * right-foot      (12)
"""
import numpy as np


def transform_from_mscoco(humans):
    """
        Transforms the mscoco detection to our structure
    :param humans: [ [(x,y,z), ..., ], ...]
    :return:
    """
    n = len(humans)
    humans_transformed = np.zeros((n, 13, 4))
    for pid, human in enumerate(humans):
        assert len(human) == 18  # mscoco standard

        # we drop the head as our data and the mscoco are
        # not compatible -> mscoco uses eyes, ears and nose
        # while we use only the tip of the head!
        mapping = [(0, 1),
                   (1, 5), (2, 6), (3, 7), (4, 2), (5, 3), (6, 4),  # arms
                   (7, 11), (8, 12), (9, 13), (10, 8), (11, 9), (12, 10)]

        for a, b in mapping:
            pt = human[b]
            if pt is not None:
                humans_transformed[pid, a, 0:3] = pt
                humans_transformed[pid, a, 3] = 1

    return np.array(humans_transformed)


def normalize(indv, settings):
    """
        normalize the data to be between -1 and 1
    :param indv:
    :param settings:
    :return:
    """
    result = indv.copy()
    n_points, n_dims = indv.shape
    assert n_points == 13
    assert n_dims == 4
    pts = indv[:, 0:3]
    visible = indv[:, 3]
    div = 1000 / settings.scale_to_mm
    mu = np.mean(pts, axis=0)
    result[:, 0:3] = (pts - mu)/div  # because we use mm!
    for i, v in enumerate(visible):
        if v == 0:
            result[i, 0] = 0
            result[i, 1] = 0
            result[i, 2] = 0
    return result


def denormalize(indv, settings):
    """
        de-normalizes the data again
    :param indv:
    :param settings:
    :return:
    """
    div = 1000 / settings.scale_to_mm
    mu = np.mean(indv, axis=0)
    return (indv + mu) * div


def plot_indv(ax, indv, visible=np.ones((13,)), color='red'):
    """
        plots the individual onto the axis
    :param ax:
    :param indv:
    :param visible:
    :param color:
    :return:
    """
    for (x, y, z), v in zip(indv, visible):
        if v == 1:
            ax.scatter(x, y, z, color=color)

    limbs = np.array([
        (1, 2), (1, 5), (2, 5),
        (2, 3), (3, 4), (5, 6), (6, 7),
        (2, 8), (5, 11), (8, 11),
        (8, 9), (9, 10), (11, 12), (12, 13)
    ]) - 1
    for a, b in limbs:
        if visible[a] == 1 and visible[b] == 1:
            p_a = indv[a]
            p_b = indv[b]
            ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], [p_a[2], p_b[2]],
                    color=color, alpha=0.8)
