import numpy as np
from mvpose.baseline.hypothesis import Hypothesis
from scipy.optimize import linear_sum_assignment


def estimate(calib, poses, epi_threshold=40,
             get_hypothesis=False):
    """
    :param calib:
    :param poses:
    :param epi_threshold:
    :return:
    """

    # -- debug<
    # lookup = {}
    # reverse_lookup = {}
    # global_id = 0
    # for cid, poses_on_image in enumerate(poses):
    #     for pid, pose in enumerate(poses_on_image):
    #         lookup[cid, pid] = global_id
    #         reverse_lookup[global_id] = (cid, pid)
    #         global_id += 1
    # -- >debug

    n_cameras = len(calib)
    assert n_cameras == len(poses)
    assert n_cameras >= 3, 'Expect multi-camera setup'

    # add all detections in the first frames as hypothesis
    # TODO: handle the case when there is NO pose in 1. cam
    first_cid = 0
    H = [
        Hypothesis(pose, calib[0], epi_threshold,
                   debug_2d_id=(first_cid, pid))
        for pid, pose in enumerate(poses[first_cid])]

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

        handled_pids = set()
        for hid, pid in zip(rows, cols):
            is_masked = Mask[hid, pid] == 1
            handled_pids.add(pid)
            if is_masked:
                # even the closest other person is
                # too far away (> threshold)
                H.append(Hypothesis(
                    all_detections[pid],
                    cam,
                    epi_threshold,
                    debug_2d_id=(cid, pid)))
            else:
                H[hid].merge(all_detections[pid], cam)
                H[hid].debug_2d_ids.append((cid, pid))

        for pid, person in enumerate(all_detections):
            if pid not in handled_pids:
                H.append(Hypothesis(
                    all_detections[pid],
                    cam,
                    epi_threshold,
                    debug_2d_id=(cid, pid)
                ))

    surviving_H = []
    humans = []
    for hyp in H:
        if hyp.size() > 1:
            humans.append(hyp.get_3d_person())
            surviving_H.append(hyp)

    if get_hypothesis:
        return humans, surviving_H
    else:
        return humans
