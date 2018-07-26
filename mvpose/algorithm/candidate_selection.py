import numpy as np
import numpy.linalg as la


def project_human_to_2d(human3d, cam):
    """

    :param human3d: [ (x,y,z) ... ] OR [ (x,y,z,w), ... ]
    :param cam:
    :return:
    """
    human2d = [None] * len(human3d)
    for jid, pt3d in enumerate(human3d):
        if pt3d is not None:
            if len(pt3d) == 3:
                Pt = np.array([pt3d])
            elif len(pt3d) == 4:
                Pt = np.array([pt3d[0:3]])
            else:
                raise ValueError('pt3d invalid:', pt3d.shape)
            points2d = cam.projectPoints_undist(Pt)  # TODO shouldn't this be WITH distortion?
            human2d[jid] = np.squeeze(points2d)
            if len(pt3d) == 4:
                human2d[jid] = np.append(human2d[jid], pt3d[3])

    return human2d


def calculate2d_proximity(person1, person2):
    n_joints = len(person1)
    assert n_joints == len(person2)
    jointwise_proximity = [-1] * n_joints

    for jid, (pt1, pt2) in enumerate(zip(person1, person2)):
        if pt1 is not None and pt2 is not None:
            distance = la.norm(pt1 - pt2)
            jointwise_proximity[jid] = distance
    return jointwise_proximity


def center_distance(human_a, human_b):
    """
        calculates the center distance
    :param human_a: 2d human
    :param human_b:
    :return:
    """
    assert len(human_a) == len(human_b)
    avg_a = []
    avg_b = []
    for i in range(len(human_a)):
        if human_a[i] is not None:
            avg_a.append(human_a[i])
        if human_b[i] is not None:
            avg_b.append(human_b[i])
    avg_a = np.mean(avg_a, axis=0)
    avg_b = np.mean(avg_b, axis=0)
    return la.norm(avg_a - avg_b)


def get_aabb(human2d):
    min_x = -1
    min_y = -1
    max_x = -1
    max_y = -1
    for jid, pt2d in enumerate(human2d):
        if pt2d is None:
            continue

        x, y = pt2d
        if min_x == -1:
            min_x = x
        if max_x == 1:
            max_x = x
        if min_y == -1:
            min_y = y
        if max_y == -1:
            max_y = y

        if min_x > x:
            min_x = x
        elif max_x < x:
            max_x = x

        if min_y > y:
            min_y = y
        elif max_y < y:
            max_y = y

    assert max_x > min_x
    assert max_y > min_y
    w = max_x - min_x
    h = max_y - min_y
    return min_x, min_y, w, h


def total_covering(aabb1, aabb2):
    I = aabb.intersection(aabb1, aabb2)
    A1 = aabb1[2] * aabb1[3]
    A2 = aabb2[2] * aabb2[3]
    cov1 = I/A1
    cov2 = I/A2
    return max(cov1, cov2)


def are_in_conflict(human2d_a, human2d_b,
                    threshold_close_pair,
                    min_nbr_joints,
                    conflict_IoU):
    """

    :param human2d_a:
    :param human2d_b:
    :param threshold_close_pair:
    :param min_nbr_joints:
    :return:
    """
    distance = calculate2d_proximity(human2d_a, human2d_b)
    count_close_pairs = 0
    for d in distance:
        if d < 0:
            continue
        if d < threshold_close_pair:
            count_close_pairs += 1

    if count_close_pairs >= min_nbr_joints:
        return True
    else:
        distance = center_distance(human2d_a, human2d_b)
        if distance < threshold_close_pair:
            return True

        aabb1 = get_aabb(human2d_a)
        aabb2 = get_aabb(human2d_b)
        IoU = aabb.IoU(aabb1, aabb2)
        if IoU >= conflict_IoU:
            return True

        if total_covering(aabb1, aabb2) >= conflict_IoU:
            return True

    return False


class CandidateSelector:

    def __init__(self, Humans, Heatmaps, Calib,
                 min_nbr_joints,
                 hm_detection_threshold,
                 threshold_close_pair):
        """
        :param Humans: 3d human candidates
        :param Heatmaps:
        :param hm_detection_threshold: threshold after which a
            detection in the confidence map is considered or not
        :param threshold_close_pair: {int} distance in pixels
            after which two points are considered to be close
        """

        # step 1: kick out ALL candidates who are not visible
        # in at least two views!
        Triangulatedable_Humans = []
        for human3d in Humans:
            n_joints = len(human3d)
            valid_views = 0
            for cid, cam in enumerate(Calib):
                human2d = project_human_to_2d(human3d, cam)
                hm = Heatmaps[cid]
                h, w, _ = hm.shape
                believe = [-1] * n_joints
                for jid, pt2d in enumerate(human2d):
                    if pt2d is not None:
                        if len(pt2d.shape) > 1:
                            pt2d = np.squeeze(pt2d)
                        x, y = np.around(pt2d).astype('int32')
                        if x > 0 and x < w and y > 0 and y < h:
                            score = hm[y, x, jid]
                            believe[jid] = score
                total = np.sum((np.array(believe) > hm_detection_threshold))
                if total >= min_nbr_joints:
                    valid_views += 1

            print('\t\t\tvalid views: #', valid_views)
            if valid_views > 1:
                Triangulatedable_Humans.append(human3d)

        Humans = Triangulatedable_Humans

        n = len(Humans)
        n_cams = len(Calib)

        # -1 -> not visible
        #  1 -> visible
        #  2 -> collision
        FLAG_NOT_VISIBLE = -1
        FLAG_VISIBLE = 1
        FLAG_COLLISION = 2
        Visibility_Table = np.zeros((n_cams, n))

        for a in range(n):
            human3d_a = Humans[a]
            n_joints = len(human3d_a)
            for cid, cam in enumerate(Calib):
                human2d_a = project_human_to_2d(human3d_a, cam)
                # ==============================================
                # check in how many views two persons co-incide
                # ==============================================
                for b in range(a + 1, n):
                    human3d_b = Humans[b]
                    human2d_b = project_human_to_2d(human3d_b, cam)

                    # check if coincidence
                    distance = calculate2d_proximity(human2d_a, human2d_b)
                    count_close_pairs = 0
                    for d in distance:
                        if d < 0:
                            continue
                        if d < threshold_close_pair:
                            count_close_pairs += 1

                    if count_close_pairs >= min_nbr_joints:
                        Visibility_Table[cid, a] = FLAG_COLLISION
                        Visibility_Table[cid, b] = FLAG_COLLISION
                    else:
                        # check if the center points are too close to each other
                        distance = center_distance(human2d_a, human2d_b)
                        if distance < threshold_close_pair:  # in [pixel]
                            Visibility_Table[cid, a] = FLAG_COLLISION
                            Visibility_Table[cid, b] = FLAG_COLLISION

                # ==============================================
                # check the heatmap values in all views
                # ==============================================
                hm = Heatmaps[cid]
                h, w, _ = hm.shape
                believe = [-1] * n_joints
                for jid, pt2d in enumerate(human2d_a):
                    if pt2d is not None:
                        if len(pt2d.shape) > 1:
                            pt2d = np.squeeze(pt2d)
                        x, y = np.around(pt2d).astype('int32')
                        if x > 0 and x < w and y > 0 and y < h:
                            score = hm[y, x, jid]
                            believe[jid] = score

                total = np.sum((np.array(believe) > hm_detection_threshold))
                if total >= min_nbr_joints:
                    Visibility_Table[cid, a] = max(FLAG_VISIBLE, Visibility_Table[cid, a])
                else:
                    Visibility_Table[cid, a] = FLAG_NOT_VISIBLE

        Valid_Humans = []
        for human3d, visibility_in_cams in zip(Humans, np.transpose(Visibility_Table)):
            valid_cams = np.sum(visibility_in_cams == 1)
            if valid_cams > 1:  # valid in at least 2 views
                Valid_Humans.append(human3d)

        self.persons = Valid_Humans


# ==============================================
# SMART CANDIDATE SELECTION
# ==============================================
from ortools.linear_solver import pywraplp as mip
import networkx as nx
from pppr import aabb


class SmartCandidateSelector:

    def __init__(self, Humans, Heatmaps, Calib,
                 min_nbr_joints,
                 conflict_IoU,
                 hm_detection_threshold,
                 threshold_close_pair,
                 debug=False):
        """
        :param Humans: 3d human candidates
        :param Heatmaps:
        :param conflict_IoU
        :param hm_detection_threshold: threshold after which a
            detection in the confidence map is considered or not
        :param threshold_close_pair: {int} distance in pixels
            after which two points are considered to be close
        """
        # step 0: get rid of all humans that are do not have enough limbs
        Survivors = []
        for human in Humans:
            n_valid_joints = 0
            for joint in human:
                if joint is not None:
                    n_valid_joints += 1
            if n_valid_joints >= min_nbr_joints:
                Survivors.append(human)
        self.All_Candidates = Humans
        self.Surviving_Candidates = Survivors
        Humans = Survivors

        G_valid = nx.Graph()
        G_conflict = nx.Graph()
        self.G_valid = G_valid
        self.G_conflict = G_conflict
        lookup = {}  # [pid, cid] -> gid
        reverse_lookup = {}  # gid -> pid, cid
        self.lookup = lookup
        self.reverse_lookup = reverse_lookup
        V = [0] * (len(Humans) * len(Calib)) # unary terms
        current_id = 0
        for pid, human in enumerate(Humans):
            for cid, cam in enumerate(Calib):
                hm = Heatmaps[cid]
                h, w, _ = hm.shape
                lookup[pid, cid] = current_id
                reverse_lookup[current_id] = (pid, cid)
                G_valid.add_node(current_id)
                G_conflict.add_node(current_id)

                human2d = project_human_to_2d(human, cam)
                unary_term = 0
                n_joints = len(human2d)
                valid_joints = 0
                for jid, pt2d in enumerate(human2d):
                    if pt2d is None:
                        continue
                    x, y = np.around(pt2d).astype('int32')
                    if x > 0 and x < w and y > 0 and y < h:
                        value = hm[y, x, jid]
                        if value < hm_detection_threshold:
                            value = 0
                        else:
                            valid_joints += 1
                        unary_term += value

                if valid_joints < min_nbr_joints:
                    unary_term = 0
                unary_term = unary_term/n_joints  # normalize to 1
                V[current_id] = unary_term
                current_id += 1

        # -- create valid edges --
        for pid in range(len(Humans)):
            for cid1 in range(len(Calib) - 1):
                for cid2 in range(cid1 + 1, len(Calib)):
                    nid1 = lookup[pid, cid1]
                    nid2 = lookup[pid, cid2]
                    if V[nid1] > 0 and V[nid2] > 0:
                        G_valid.add_edge(nid1, nid2)

        # -- create conflict edges --
        for pid, human in enumerate(Humans):
            all_candidate_ids = []  # all ids for this candidate
            for cid, cam in enumerate(Calib):
                # check for conflicts
                human2d = project_human_to_2d(human, cam)
                for pid2 in range(pid + 1, len(Humans)):
                    human2d_b = project_human_to_2d(Humans[pid2], cam)

                    if are_in_conflict(human2d, human2d_b,
                                       threshold_close_pair,
                                       min_nbr_joints,
                                       conflict_IoU):
                        nid1 = lookup[pid, cid]
                        nid2 = lookup[pid2, cid]
                        G_conflict.add_edge(nid1, nid2)

        # ==================================
        # optimization
        # ==================================
        solver = mip.Solver('cand', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        n = len(V)
        Sum = []
        Pi = {}
        Rho = {}
        Chi = {}
        for idx in range(n):
            Rho[idx] = solver.BoolVar('rho[%i]' % idx)

        s = solver.Sum(Rho[idx] * V[idx] for idx in range(n))
        Sum.append(s)

        # --- constraints for conflicts ---
        for clique in nx.find_cliques(G_conflict):
            # ~~~~ (1) ~~~~
            solver.Add(
                solver.Sum(Rho[node] for node in clique) <= 1)

        # --- constraints for valid edges ---
        for a, b in G_valid.edges():
            Chi[a, b] = solver.BoolVar('chi[%i,%i' % (a, b))
            # ~~~~ (2) ~~~~
            solver.Add(2 * Chi[a, b] <= Rho[a] + Rho[b])

        s = solver.Sum(Chi[a, b] * 1 for (a, b) in G_valid.edges())

        for pid, clique in enumerate(nx.connected_components(G_valid)):
            Pi[pid] = solver.BoolVar('pi[%i]' % pid)
            # ~~~~ (3) ~~~~
            solver.Add(solver.Sum(Rho[a] for a in clique) <= Pi[pid] * n)

            # ~~~~ (4) ~~~~
            solver.Add(solver.Sum(Rho[a] for a in clique) >= 2 * Pi[pid])


        Sum.append(s)
        solver.Maximize(solver.Sum(Sum))
        RESULT = solver.Solve()
        if debug:
            print('(smart candidate selection) [')
            print("\tTime = ", solver.WallTime(), " ms")
            print("\tresult:", RESULT)
            print('\n\tTotal cost:', solver.Objective().Value())
            print('] (smart candidate selection)')

        self.persons = []

        # check which persons 'survive'
        for pid, human in enumerate(Humans):
            nbr_survivors = 0
            for cid in range(len(Calib)):
                nid = lookup[pid, cid]
                if Rho[nid].solution_value() > 0:
                    nbr_survivors += 1
            if nbr_survivors > 1:
                self.persons.append(human)
