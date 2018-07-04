from ortools.linear_solver import pywraplp as mip
import numpy.linalg as la
import numpy as np
from time import time
from mvpose.geometry import geometry as gm


def distance3d_humans(human1, human2):
    """
        calculates the distance between two humans
    :param human1: [ (x, y, z, ?), ... ]
    :param human2:  -"-
    :return:
    """
    h1 = []
    h2 = []
    for j1, j2 in zip(human1, human2):
        if j1 is not None:
            h1.append(j1[0:3])
        if j2 is not None:
            h2.append(j2[0:3])
    p1 = np.mean(h1, axis=0)
    p2 = np.mean(h2, axis=0)
    return la.norm(p1 - p2)


def get_bb(cam, human, w, h):
    """
        gets the aabb from the human using the camera
    :param cam:
    :param human:
    :param w: image width
    :param h: image height
    :return:
    """
    points3d = []
    for joints in human:
        if joints is not None:
            points3d.append(joints)
    points2d = cam.projectPoints(np.array(points3d))
    points2d = np.round(points2d)
    max_x = np.clip(int(np.max(points2d[:, 0])), a_max=w-1, a_min=0)
    min_x = np.clip(int(np.min(points2d[:, 0])), a_max=w-1, a_min=0)
    max_y = np.clip(int(np.max(points2d[:, 1])), a_max=h-1, a_min=0)
    min_y = np.clip(int(np.min(points2d[:, 1])), a_max=h-1, a_min=0)
    return min_x, min_y, max_x, max_y


class GraphPartitioningTracker:

    def __init__(self, Calibs, Ims, humans_candidates, debug,
                 tracking_setting):
        """
            Tracking using human poses
        :param Calibs:
        :param Ims:
        :param humans_candidates:
        :param valid_person_bb_area: valid area in [pixel] over
            which a person reprojection into an image is considered
            valid
        :param tracking_setting: {mvpose.algorithm.settings.Tracking_Settings}
        :param debug
        """
        n_frames = len(Ims)
        assert len(humans_candidates) == n_frames
        assert len(Calibs) == n_frames
        low_spec_mode = tracking_setting.low_spec_mode

        # =====================================
        # calculate Edge costs
        # =====================================
        valid_person_bb_area = tracking_setting.valid_person_bb_area
        max_moving_distance = tracking_setting.max_moving_distance_per_frame
        moving_factor_increase = tracking_setting.moving_factor_increase_per_frame
        conflict_IoU = tracking_setting.conflict_IoU
        _, _, H, W, _ = Ims.shape

        _start = time()
        # we first want to figure out which person reprojections we can safely
        # use and which ones we cannot as others are in the vicinity
        # (t, cid, pid)
        conflict_lookup = set()  # if an item is "set", there is a conflict
        for t in range(n_frames):
            n_humans = len(humans_candidates[t])
            for cid, cam in enumerate(Calibs[t]):
                for pid, human in enumerate(humans_candidates[t]):
                    aabb = get_bb(cam, human, W, H)

                    # if the candidate is too small in this particular camera we
                    # simply ignore it
                    if gm.aabb_area(aabb) < valid_person_bb_area:
                        conflict_lookup.add((t, cid, pid))
                        continue

                    # loop over all other humans in this particular view and check if
                    # there is a conflict
                    for pid2 in range(pid + 1, n_humans):
                        aabb2 = get_bb(cam, humans_candidates[t][pid2], W, H)
                        if gm.aabb_IoU(aabb, aabb2) > conflict_IoU:
                            conflict_lookup.add((t, cid, pid))
                            conflict_lookup.add((t, cid, pid2))
                            break  # one conflict per pid is enough

        ImgsA = []
        ImgsB = []
        pairs = []  # t1, pid1, cid1, t2 pid2, cid2
        for t1 in range(n_frames - 1):
            for t2 in range(t1 + 1, n_frames):
                dt = t2 - t1
                max_d = max_moving_distance * dt * moving_factor_increase
                for pidA, candA in enumerate(humans_candidates[t1]):
                    for pidB, candB in enumerate(humans_candidates[t2]):
                        # -- when the two person candidates are too far
                        # -- away from each other (depending on dt) we
                        # -- will not make them linkable candidates
                        distance = distance3d_humans(candA, candB)
                        if distance > max_d:
                            continue

                        for cidA, camA in enumerate(Calibs[t1]):
                            if (t1, cidA, pidA) in conflict_lookup:
                                # in case of a conflicting camera view we
                                # simply skip to the next one
                                continue

                            for cidB, camB in enumerate(Calibs[t2]):
                                if (t2, cidB, pidB) in conflict_lookup:
                                    # in case of a conflicting camera view we
                                    # simply skip to the next one
                                    continue

                                aabb_A = get_bb(camA, candA, W, H)
                                aabb_B = get_bb(camB, candB, W, H)
                                # if gm.aabb_area(aabb_A) < valid_person_bb_area \
                                #         or gm.aabb_area(aabb_B) < valid_person_bb_area:
                                #     # the bounding box must have a minimal area in the
                                #     # camera view to be considered
                                #     continue

                                tx, ty, bx, by = aabb_A
                                imga = Ims[t1][cidA][ty: by, tx: bx]
                                ImgsA.append(imga)
                                tx, ty, bx, by = aabb_B
                                imgb = Ims[t2][cidB][ty: by, tx: bx]
                                ImgsB.append(imgb)
                                pairs.append((t1, pidA, cidA,
                                             t2, pidB, cidB))
        _end = time()

        self.ImgsA = ImgsA
        self.ImgsB = ImgsB
        self.pairs = pairs
        if debug:
            print('\t[gp:step 1] elapsed', _end - _start)
            print('\t\t# boxes to compare:', len(ImgsA))

        # =====================================
        # predict similarity
        # =====================================
        _start = time()
        if low_spec_mode:
            scores = np.zeros((0,))
            batchsize = tracking_setting.personreid_batchsize
            for i in range(0, len(ImgsB), batchsize):
                A = ImgsA[i: i + batchsize]
                B = ImgsB[i: i + batchsize]
                _scores = tracking_setting.reid_model.predict(A, B)
                scores = np.concatenate([scores, _scores], axis=0)
        else:
            scores = tracking_setting.reid_model.predict(ImgsA, ImgsB)
        _end = time()
        self.scores = scores
        assert len(scores) == len(ImgsA)

        if debug:
            print('\t[gp:step 2] elapsed', _end - _start)

        # =====================================
        # calculate 3d scores
        # =====================================
        _start = time()

        div = {}
        graph_3d = {}  # tA, pidA, tB, pidB
        for s, meta in zip(scores, pairs):
            tA, pidA, _, tB, pidB, _ = meta
            assert tA < tB
            if not (tA, pidA, tB, pidB) in graph_3d:
                graph_3d[tA, pidA, tB, pidB] = s
                div[tA, pidA, tB, pidB] = 1
            else:
                graph_3d[tA, pidA, tB, pidB] += s
                div[tA, pidA, tB, pidB] += 1
        for key, value in div.items():
            graph_3d[key] = graph_3d[key]/value

        self.graph_3d = graph_3d
        _end = time()
        if debug:
            print('\t[gp:step 3] elapsed', _end - _start)
            print("\t\t# nodes in 3D graph:", len(graph_3d))

        # =====================================
        # build graph
        # =====================================
        solver = mip.Solver('t', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        pids_per_frame = {}
        Tau = {}
        costs = {}
        for (tA, pidA, tB, pidB), score in graph_3d.items():
            Tau[tA, pidA, tB, pidB] = solver.BoolVar('t[%i,%i,%i,%i]' % (tA, pidA, tB, pidB))
            costs[tA, pidA, tB, pidB] = np.log(score / (1-score))
            if tA not in pids_per_frame:
                pids_per_frame[tA] = []
            pids_per_frame[tA].append(pidA)
            if tB not in pids_per_frame:
                pids_per_frame[tB] = []
            pids_per_frame[tB].append(pidB)

        self.costs = costs
        Sum = solver.Sum(Tau[edge] * costs[edge] for edge in graph_3d.keys())

        # -- add constraints --
        for t1 in range(n_frames - 1):
            for t2 in range(t1 + 1, n_frames):
                solver.Add(
                    solver.Sum(Tau[t1, pid1, t2, pid2]\
                               for pid1 in pids_per_frame[t1]\
                               for pid2 in pids_per_frame[t2] if (t1, pid1, t2, pid2) in Tau) <= 1
                )

        solver.Maximize(Sum)
        RESULT = solver.Solve()
        if debug:
            print('-------------------------------------------')
            print("\t\tTime = ", solver.WallTime(), " ms")
            print("\t\tresult:", RESULT)
            print('\n\t\tTotal cost:', solver.Objective().Value())

        # =====================================
        # extract tracks
        # =====================================
        for (t1, pid1, t2, pid2), v in Tau.items():
            if v.solution_value() > 0:
                pass
