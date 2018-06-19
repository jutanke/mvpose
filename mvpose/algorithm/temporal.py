import numpy as np
import numpy.linalg as la


def avg_distance(human1, human2):
    """
        calculates avg distance between two humans
    :param human1:
    :param human2:
    :return:
    """
    n_joints = len(human1)
    assert n_joints == len(human2)
    all_distances = []
    for jid in range(n_joints):
        pt1 = human1[jid]
        pt2 = human2[jid]
        assert len(pt1) == 3
        assert len(pt2) == 3
        if pt1 is not None and pt2 is not None:
            d = la.norm(pt1 - pt2)
            all_distances.append(d)
    return np.mean(all_distances)