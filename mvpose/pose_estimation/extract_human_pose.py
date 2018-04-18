from scipy.optimize import linear_sum_assignment


def extract(modes, W, limbSeq, threshold_drop_person=10):
    """
        extracts the human poses from the items given by
        the candidate generation
    :param modes: [ (nx3), ... x m]
    :param W: [ (nxp)  ... x m]
    :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
    :param threshold_drop_person: {integer} if less then this threshold
                        items are found: drop the whole detection
    :return:
    """
    pid = 0

    # represents all modes and their respetive pid (-1 => no person)
    modes_to_person = [[-1] * len(x) for x in modes]

    for lid, (k1, k2) in enumerate(limbSeq):
        # p1 = modes[k1]; p2 = modes[k2]
        lw = -W[lid]

        rows, cols = linear_sum_assignment(lw)

        for a, b in zip(rows, cols):
            v1 = modes_to_person[k1][a]
            v2 = modes_to_person[k2][b]

            if v1 <= 0 and v2 <= 0:  # no person is set: set a new pid
                modes_to_person[k1][a] = pid
                modes_to_person[k2][b] = pid
                pid += 1
            elif v1 <= 0 and v2 >= 0:
                modes_to_person[k1][a] = v2
            elif v2 <= 0 and v1 >= 0:
                modes_to_person[k2][b] = v1
            else:  # merge people
                pass  # TODO we need to do something here..
    #             for i in range(len(modes_to_person)):
    #                 for j in range(len(modes_to_person[i])):
    #                     if modes_to_person[i][j] == v2:
    #                         modes_to_person[i][j] = v1

    # remove detections with too few points
    # TODO make this more efficient...
    count_pids = [0] * pid

    for pids in modes_to_person:
        for pid in pids:
            if pid >= 0:
                count_pids[pid] += 1

    humans = {}
    for pid, count in enumerate(count_pids):
        if count > threshold_drop_person:
            humans[pid] = [None] * len(modes_to_person)

    for k, pids in enumerate(modes_to_person):
        for idx, pid in enumerate(pids):
            if pid in humans:
                humans[pid][k] = modes[k][idx]

    return humans
