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
    avg_loc1 = []
    avg_loc2 = []
    for jid in range(n_joints):
        pt1 = human1[jid]
        pt2 = human2[jid]
        if pt1 is not None:
            assert len(pt1) == 3
            avg_loc1.append(pt1)
        if pt2 is not None:
            assert len(pt2) == 3
            avg_loc2.append(pt2)

    avg_loc1 = np.mean(avg_loc1)
    avg_loc2 = np.mean(avg_loc2)
    return la.norm(avg_loc2 - avg_loc1)
