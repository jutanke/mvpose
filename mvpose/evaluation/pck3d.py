import numpy.linalg as la
import numpy as np


def evaluate(gt, d, scale_to_mm, maxdistance=150):
    """
        percentage of correctly estimated parts.
        This score only works on single-human estimations
        and the 3d data must be transformed to fit the
        KTH football2 data format (see {transform3d_from_mscoco})
    :param gt: ground truth human
    :param d: detection human
    :param alpha: 0.5
    :return:
    """
    assert len(gt) == 14
    result = np.zeros((14,), np.int64)
    if d is not None:
        assert len(d) == 14
        for jid in range(14):
            pt_gt = gt[jid]
            pt_pr = d[jid]
            if pt_pr is not None:
                distance = la.norm(pt_gt - pt_pr) * scale_to_mm
                if distance <= maxdistance:
                    result[jid] = 1
    return np.mean(result)
