import numpy as np
from ortools.linear_solver import pywraplp as mip
from mvpose.algorithm.transitivity import TransitivityLookup
import mvpose.geometry.geometry as gm
import networkx as nx


def merge(person):
    result = [None] * len(person)
    for jid, points3d in enumerate(person):
        if points3d is not None:
            Pos = points3d[:, 0:3]
            unary = get_unary(points3d)
            W = np.expand_dims(unary, axis=1)
            S = np.sum(W)
            WPos = np.multiply(Pos, W)
            WPos = np.sum(WPos, axis=0) / S
            result[jid] = WPos
    return result


def get_unary(pts3d):
    """

    :param pts3d:
    :return:
    """
    if pts3d.shape[1] == 5:
        left = pts3d[:, 3]
        right = pts3d[:, 4]
        unary = np.multiply(left, right)
    elif pts3d.shape[1] == 4:
        unary = pts3d[:, 3]
    else:
        raise ValueError("Shape of Points3d is wrong", pts3d.shape)
    return unary


class Graphcut:

    def __init__(self, params, points3d, limbs3d, debug=False):
        """

        :param params:
        :param points3d: {np.array} [ (x,y,z,w1,w2), ... ]
        :param limbs3d:
        :param debug:
        """
        limbSeq = params.limb_seq
        sensible_limb_length = params.sensible_limb_length
        scale_to_mm = params.scale_to_mm
        max_radius = params.gc_max_radius
        radius = params.gc_radius
        min_nbr_joints = params.min_nbr_joints
        iota_scale = params.gc_iota_scale
        sensible_limb_length = sensible_limb_length
        n_joints = len(points3d)

        # ===========================================
        # COST  FUNCTIONS
        # ===========================================
        pboost_big = lambda x: np.log((x + 1) / (2 * (0.5 * (-x - 1) + 1))) * 2
        pboost_small = lambda x: np.log(x / (1 - x))
        # func1 = lambda u: np.tanh(pboost_small(u))
        # func2 = lambda d: (-np.tanh(((d * scale_to_mm) - radius) / radius) * iota_scale)
        # func3 = lambda x: pboost_big(x)
        func1 = lambda u: pboost_small(u)
        func2 = lambda d: (-np.tanh(((d * scale_to_mm) - radius) / radius) * iota_scale)
        func3 = lambda x: pboost_big(x)

        # ===========================================
        # CREATE COST AND BOOLEAN VARIABLES
        # ===========================================
        solver = mip.Solver('m', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        D = []  # all nodes of the graph    (jid, a)
        E_j = []  # all edges on the joints (jid, a, b)
        E_l = []  # all edges on the limbs  (jid1, jid2, a, b)

        Nu = {}
        Iota = {}
        Lambda = {}
        Get_Iota = lambda jid, a, b: Iota[jid, min(a, b), max(a, b)]
        Get_Lambda = lambda jid1, jid2, a, b: Lambda[jid1, jid2, a, b] \
            if (jid1, jid2, a, b) in Lambda else Lambda[jid2, jid1, b, a]

        Sum = []
        for jid, pts3d in enumerate(points3d):
            # ===========================================
            # HANDLE NU
            # ===========================================
            unary = get_unary(pts3d)
            unary = np.clip(unary, a_min=0.00000001, a_max=0.99999999)

            n = len(unary)
            for idx in range(n):
                Nu[jid, idx] = solver.BoolVar('nu[%i,%i]' % (jid, idx))
                D.append((jid, idx))

            s = solver.Sum(
                Nu[jid, idx] * func1(unary[idx]) for idx in range(n))
            Sum.append(s)

            # ===========================================
            # HANDLE IOTA
            # ===========================================
            # (a, b, distance)
            distance = gm.calculate_distance_all4all(
                points3d[jid], points3d[jid], max_distance=max_radius,
                AB_are_the_same=True)
            As = distance[:, 0].astype('int32')
            Bs = distance[:, 1].astype('int32')

            for a, b in zip(As, Bs):
                Iota[jid, a, b] = solver.BoolVar('j[%i,%i,%i]' % (jid, a, b))
                E_j.append((jid, a, b))

            s = solver.Sum(
                Iota[jid, int(a), int(b)] * func2(d) for a, b, d in distance)
            Sum.append(s)

        # ===========================================
        # HANDLE LAMBDA
        # ===========================================
        for lid, ((jid1, jid2), (mindist, maxdist)) in \
                enumerate(zip(limbSeq, sensible_limb_length)):
            assert jid1 != jid2
            ABdistance = gm.calculate_distance_all4all(
                points3d[jid1], points3d[jid2],
                max_distance=maxdist,
                min_distance=mindist,
                AB_are_the_same=False)
            As = ABdistance[:, 0].astype('int32')
            Bs = ABdistance[:, 1].astype('int32')

            for a, b in zip(As, Bs):
                Lambda[jid1, jid2, a, b] = solver.BoolVar(
                    'l[%i,%i,%i,%i]' % (jid1, jid2, a, b))
                E_l.append((jid1, jid2, a, b))

            W = limbs3d[lid]
            Scores = W[As, Bs]

            s = solver.Sum(
                Get_Lambda(jid1, jid2, a, b) * func3(s) for a, b, s in \
                zip(As, Bs, Scores))
            Sum.append(s)

        # ===========================================
        # ONLY CONSIDER VALID EDGES
        # ===========================================
        for jid1, jid2, a, b in E_l:
            solver.Add(
                Lambda[jid1, jid2, a, b] * 2 <= Nu[jid1, a] + Nu[jid2, b])

        for jid, a, b in E_j:
            solver.Add(
                Iota[jid, a, b] * 2 <= Nu[jid, a] + Nu[jid, b])

        # ===========================================
        # HANDLE TRANSITIVITY CONSTRAINTS (1)
        # ===========================================

        Intra = []  # [ (jid, a, b, c), ...]
        Inter = []  # [ (jid1, a, b, jid2, c), ...]
        Intra_choice = []  # [ (jid, a, b, c), ...]
        Inter_choice = []  # [ (jid1, a, jid2, b, jid3, c), ...]

        transitivity_lookup = TransitivityLookup(D, E_l, E_j)
        for q in D:
            intra, intra_choice, inter, inter_choice = \
                transitivity_lookup.query_with_choice(*q)
            Intra += intra
            Inter += inter
            Intra_choice += intra_choice
            Inter_choice += inter_choice

        assert len(Inter) == len(set(Inter))
        assert len(Intra) == len(set(Intra))
        assert len(Inter_choice) == len(set(Inter_choice))
        assert len(Intra_choice) == len(set(Intra_choice))

        if debug:
            print('-------------------------------------------')
            print('Handle transitivity:')
            print('\tIntra:\t\t', len(Intra))
            print('\tIntra(choice):\t', len(Intra_choice))
            print('\tInter:\t\t', len(Inter))
            print('\tInter(choice):\t', len(Inter_choice))

        # ===========================================
        # HANDLE TRANSITIVITY CONSTRAINTS (2)
        # ===========================================
        for jid, a, b, c in Intra:
            assert a < b < c
            solver.Add(Get_Iota(jid, a, b) + Get_Iota(jid, a, c) - 1 <= \
                       Get_Iota(jid, b, c))
            solver.Add(Get_Iota(jid, a, b) + Get_Iota(jid, b, c) - 1 <= \
                       Get_Iota(jid, a, c))
            solver.Add(Get_Iota(jid, a, c) + Get_Iota(jid, b, c) - 1 <= \
                       Get_Iota(jid, a, b))

        for jid1, a, b, jid2, c in Inter:
            solver.Add(
                Get_Lambda(jid1, jid2, a, c) + Get_Lambda(jid1, jid2, b, c) - 1 <= \
                Get_Iota(jid1, a, b))
            solver.Add(
                Get_Iota(jid1, a, b) + Get_Lambda(jid1, jid2, a, c) - 1 <= \
                Get_Lambda(jid1, jid2, b, c))
            solver.Add(
                Get_Iota(jid1, a, b) + Get_Lambda(jid1, jid2, b, c) - 1 <= \
                Get_Lambda(jid1, jid2, a, c))

        # ===========================================
        # HANDLE CHOICE CONSTRAINTS
        # ===========================================
        for jid, a, b, c in Intra_choice:  # either take { ab OR ac }
            solver.Add(
                Get_Iota(jid, a, b) + Get_Iota(jid, a, c) <= 1
            )

        for jid1, a, jid2, b, jid3, c in Inter_choice:  # { ab OR ac }
            if jid1 == jid2:
                assert jid3 != jid1
                # if  [a]---[b]
                #     |
                #    (c)
                solver.Add(
                    Get_Iota(jid1, a, b) + Get_Lambda(jid1, jid3, a, c) <= 1
                )
            elif jid2 == jid3:
                # if  [a]
                #     |   \
                #    (b)   (c)
                solver.Add(
                    Get_Lambda(jid1, jid2, a, b) + Get_Lambda(jid1, jid3, a, c) <= 1
                )
            elif jid1 == jid3:
                # if  [a]---[c]
                #     |
                #    (b)
                solver.Add(
                    Get_Lambda(jid1, jid2, a, b) + Get_Iota(jid1, a, c) <= 1
                )
            else:
                raise ValueError("nope")

        # ===========================================
        # SOLVE THE GRAPH
        # ===========================================
        solver.Maximize(solver.Sum(Sum))
        RESULT = solver.Solve()
        if debug:
            print('-------------------------------------------')
            print("\tTime = ", solver.WallTime(), " ms")
            print("\tresult:", RESULT)
            print('\n\tTotal cost:', solver.Objective().Value())

        # ===========================================
        # EXTRACT INDIVIDUALS
        # ===========================================

        valid_3d_points = set()
        count_invalid_points = 0
        for (jid, idx), v in Nu.items():
            if v.solution_value() > 0:
                valid_3d_points.add((jid, idx))
            else:
                count_invalid_points += 1
        if debug:
            print("\n# valid points:\t\t", len(valid_3d_points))
            print("# invalid points:\t", count_invalid_points)

        G = nx.Graph()
        for (jid1, jid2, a, b), v in Lambda.items():
            if v.solution_value() > 0:
                assert (jid1, a) in valid_3d_points and (jid2, b) in valid_3d_points
                candA = transitivity_lookup.lookup[jid1, a]
                candB = transitivity_lookup.lookup[jid2, b]
                G.add_edge(candA, candB)

        for (jid, a, b), v in Iota.items():
            if v.solution_value() > 0:
                assert (jid, a) in valid_3d_points and (jid, b) in valid_3d_points
                candA = transitivity_lookup.lookup[jid, a]
                candB = transitivity_lookup.lookup[jid, b]
                G.add_edge(candA, candB)

        persons = []
        for comp in nx.connected_components(G):
            person = [None] * n_joints
            for node in comp:
                jid, idx = transitivity_lookup.reverse_lookup[node]
                if person[jid] is None:
                    person[jid] = []
                person[jid].append(points3d[jid][idx])
            persons.append(person)

        for pid in range(len(persons)):
            for jid in range(n_joints):
                if persons[pid][jid] is not None:
                    persons[pid][jid] = np.array(persons[pid][jid])

        valid_persons = []
        for pid, person in enumerate(persons):
            count_valid_joints = 0
            for jid in range(n_joints):
                if persons[pid][jid] is not None:
                    count_valid_joints += 1

            if count_valid_joints >= min_nbr_joints:
                valid_persons.append(person)

        assert len(Nu) == len(D)
        assert len(E_j) == len(Iota)
        assert len(E_l) == len(Lambda)

        Humans = [merge(p) for p in valid_persons]

        self.person_candidates_all = [merge(p) for p in persons]
        self.person_candidates = Humans