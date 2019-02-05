import numpy as np
import numpy.linalg as la
from mvpose.baseline.hypothesis import Hypothesis, HypothesisList
from scipy.optimize import linear_sum_assignment


def distance_between_poses(pose1, pose2):
    """
    :param pose1:
    :param pose2:
    :return:
    """
    J = len(pose1)
    assert len(pose2) == J
    distances = []
    for jid in range(J):
        if pose2[jid] is None or pose1[jid] is None:
            continue
        d = la.norm(pose2[jid] - pose1[jid])
        distances.append(d)

    if len(distances) == 0:
        # TODO check this heuristic
        # take the centre distance in x-y coordinates
        valid1 = []
        valid2 = []
        for jid in range(J):
            if pose1[jid] is not None:
                valid1.append(pose1[jid])
            if pose2[jid] is not None:
                valid2.append(pose2[jid])

        assert len(valid1) > 0
        assert len(valid2) > 0
        mean1 = np.mean(valid1, axis=0)
        mean2 = np.mean(valid2, axis=0)
        assert len(mean1) == 3
        assert len(mean2) == 3

        # we only care about xy coordinates
        mean1[2] = 0
        mean2[2] = 0

        return la.norm(mean1 - mean2)
    else:
        return np.mean(distances)  # TODO try different versions


def estimate(calib, poses,
             scale_to_mm=1,
             merge_distance=-1,
             epi_threshold=40,
             variance_threshold=-1,
             get_hypothesis=False):
    """
    :param calib:
    :param poses:
    :param scale_to_mm: d * scale_to_mm = d_in_mm
        that means: if our scale is in [m] we need to set
        scale_to_mm = 1000
    :param merge_distance: in [mm]
    :param epi_threshold:
    :param get_hypothesis:
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
                   scale_to_mm=scale_to_mm,
                   variance_threshold=variance_threshold,
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
                    scale_to_mm=scale_to_mm,
                    variance_threshold=variance_threshold,
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
                    scale_to_mm=scale_to_mm,
                    variance_threshold=variance_threshold,
                    debug_2d_id=(cid, pid)))

    surviving_H = []
    humans = []
    for hyp in H:
        if hyp.size() > 1:
            humans.append(hyp.get_3d_person())
            surviving_H.append(hyp)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # merge closeby poses
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if merge_distance > 0:
        distances = []  # (hid1, hid2, distance)
        n = len(humans)
        for i in range(n):
            for j in range(i+1, n):
                pose1 = humans[i]
                pose2 = humans[j]
                distance = distance_between_poses(pose1, pose2)
                distances.append((i, j, distance * scale_to_mm))

        # the root merge is always the smallest hid
        # go through all merges and point higher hids
        # towards their smallest merge hid

        mergers_root = {}  # hid -> root
        mergers = {}  # root: [ hid, hid, .. ]
        all_merged_hids = set()
        for hid1, hid2, distance in distances:
            if distance > merge_distance:
                continue

            if hid1 in mergers_root and hid2 in mergers_root:
                continue  # both are already handled

            if hid1 in mergers_root:
                hid1 = mergers_root[hid1]

            if hid1 not in mergers:
                mergers[hid1] = [hid1]

            mergers[hid1].append(hid2)
            mergers_root[hid2] = hid1
            all_merged_hids.add(hid1)
            all_merged_hids.add(hid2)

        merged_surviving_H = []
        merged_humans = []

        for hid in range(n):
            if hid in mergers:
                hyp_list = [surviving_H[hid2] for hid2 in mergers[hid]]
                hyp = HypothesisList(hyp_list)
                pose = hyp.get_3d_person()
                merged_surviving_H.append(hyp)
                merged_humans.append(pose)
            elif hid not in all_merged_hids:
                merged_surviving_H.append(surviving_H[hid])
                merged_humans.append(humans[hid])

        humans = merged_humans
        surviving_H = merged_surviving_H
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if get_hypothesis:
        return humans, surviving_H
    else:
        return humans
