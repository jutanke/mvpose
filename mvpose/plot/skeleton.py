"""
Given a set of joints, generate a skeleton
* head (1)
* neck (2)
* left-shoulder (3)
* left-elbow (4)
* left-hand (5)
* right-shoulder (6)
* right-elbow (7)
* right-hand (8)
* left-hip (9)
* left-knee (10)
* left-foot (11)
* right-hip (12)
* right-knee (13)
* right-foot (14)
"""
import numpy as np


def generate3d(ax, human, limbSeq, color, alpha=0.4):
    """

    :param human: [ (x,y,z) ... * m]
    :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
    :return:
    """

    for k1, k2 in limbSeq:
        p1 = human[k1]
        p2 = human[k2]
        if p1 is not None and p2 is not None:
            try:
                x1, y1, z1 = p1
                x2, y2, z2 = p2
            except ValueError:
                x1, y1, z1, _ = p1
                x2, y2, z2, _ = p2
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, alpha=alpha, linewidth=5)


def generate(Y):
    """ Given a set of joints, generate a skeleton
    :param Y: [14*#persons, (x,y)]
    :return:
    """
    total_joints, q = Y.shape; assert q == 2
    assert total_joints % 14 == 0, 'Y does not adhere to the mvpose standard'

    n_persons = int(total_joints / 14)

    PLOTS = []

    for pid in range(n_persons):
        x = []
        y = []

        def add(ptr):
            """ add ptr in Y to x,y
            """
            x.append(Y[ptr, 0])
            y.append(Y[ptr, 1])

        ptr = pid * 14
        add(ptr)        # head
        add(ptr + 1)    # neck
        add(ptr + 2)    # left shoulder
        add(ptr + 3)    # left elbow
        add(ptr + 4)    # left hand
        add(ptr + 3); add(ptr + 2)
        add(ptr + 5)    # right shoulder
        add(ptr + 6)    # right elbow
        add(ptr + 7)    # right hand
        add(ptr + 6); add(ptr + 5)
        add(ptr + 11)   # right hip
        add(ptr + 12)   # right knee
        add(ptr + 13)   # right foot
        add(ptr + 12); add(ptr + 11)
        add(ptr + 8)    # left hip
        add(ptr + 2); add(ptr + 8)
        add(ptr + 9)    # left knee
        add(ptr + 10)   # left foot
        PLOTS.append((x,y))

    return np.array(PLOTS)

