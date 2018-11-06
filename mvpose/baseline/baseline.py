import numpy as np
from mvpose.baseline.hypothesis import Hypothesis
from scipy.optimize import linear_sum_assignment


def estimate(calib, poses, epi_threshold=20):
    """
    :param calib:
    :param poses:
    :param epi_threshold:
    :return:
    """
    n_cameras = len(calib)
    assert n_cameras == len(poses)
    assert n_cameras >= 3, 'Expect multi-camera setup'

    # add all detections in the first frames as hypothesis
    # TODO: handle the case when there is NO pose in 1. cam
    H = [
        Hypothesis(pose, calib[0], epi_threshold)
        for pose in poses[0]]

    for cid in range(1, n_cameras):
        cam = calib[cid]
        all_detections = poses[cid]

        n_hyp = len(H)
        n_det = len(all_detections)

        C = np.zeros((n_hyp, n_det))
        Mask = np.zeros_like(C).astype('int32')

        for pid, person in enumerate(all_detections):
            for hid, h in enumerate(H):
                cost, veto = h.calculate_cost(person, cam)
                C[hid, pid] = cost
                if veto:
                    Mask[hid, pid] = 1

        rows, cols = linear_sum_assignment(C)

        for hid, pid in zip(rows, cols):
            is_masked = Mask[hid, pid] == 1
            if is_masked:
                # even the closest other person is
                # too far away (> threshold)
                H.append(Hypothesis(
                    all_detections[pid],
                    cam,
                    epi_threshold))
            else:
                H[hid].merge(all_detections[pid], cam)
