from mvpose.algorithm.graphcut import PBOOST_BIG, PBOOST_SMALL, get_unary
import numpy as np
from ortools.linear_solver import pywraplp as mip
from mvpose.algorithm.transitivity import TransitivityLookup
import mvpose.geometry.geometry as gm


class GraphcutTracking:

    def __init__(self, params, Points3d, Limbs3d, debug=False):
        """

        :param params:
        :param Points3d: {np.array} [[ (x,y,z,w1,w2), ... ]]
        :param Limbs3d:
        :param debug:
        """
        n_frames = len(Points3d)
        assert n_frames == len(Limbs3d)
        limbSeq = params.limb_seq
        sensible_limb_length = params.sensible_limb_length
        scale_to_mm = params.scale_to_mm
        max_radius = params.gc_max_radius
        radius = params.gc_radius
        min_nbr_joints = params.min_nbr_joints
        iota_scale = params.gc_iota_scale
        sensible_limb_length = sensible_limb_length
        min_symmetric_distance = params.min_symmetric_distance
        symmetric_joints = params.symmetric_joints
        n_joints = len(Points3d[0])

        # ===========================================
        # COST  FUNCTIONS
        # ===========================================
        func1 = lambda u: PBOOST_SMALL(u)
        func2 = lambda d: (-np.tanh(((d * scale_to_mm) - radius) / radius) * iota_scale)
        func3 = lambda x: PBOOST_BIG(x)

        # ===========================================
        # CREATE COST AND BOOLEAN VARIABLES
        # ===========================================
        solver = mip.Solver('m', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        Nu = {}  # frame, jid, a
        Iota = {}  # frame, jid, a, b
        Lambda = {}  # frame jid1, jid2, a, b
        Get_Iota = lambda f, jid, a, b: Iota[f, jid, min(a, b), max(a, b)]
        Get_Lambda = lambda f, jid1, jid2, a, b: Lambda[f, jid1, jid2, a, b] \
            if (f, jid1, jid2, a, b) in Lambda else Lambda[f, jid2, jid1, b, a]

        Sum = []

        for frame, (points3d, limbs3d) in enumerate(zip(Points3d, Limbs3d)):
            D = []  # all nodes of the graph    (jid, a)
            E_j = []  # all edges on the joints (jid, a, b)
            E_l = []  # all edges on the limbs  (jid1, jid2, a, b)

            for jid, pts3d in enumerate(points3d):
                # ===========================================
                # HANDLE NU
                # ===========================================
                unary = get_unary(pts3d)
                unary = np.clip(unary, a_min=0.00000001, a_max=0.99999999)

                n = len(unary)
                for idx in range(n):
                    Nu[frame, jid, idx] = solver.BoolVar('nu[%i,%i,%i]' % (frame, jid, idx))
                    D.append((jid, idx))

                s = solver.Sum(
                    Nu[frame, jid, idx] * func1(unary[idx]) for idx in range(n))
                Sum.append(s)

                # ===========================================
                # HANDLE IOTA
                # ===========================================
                distance = gm.calculate_distance_all4all(
                    points3d[jid], points3d[jid], max_distance=max_radius,
                    AB_are_the_same=True)
                As = distance[:, 0].astype('int32')
                Bs = distance[:, 1].astype('int32')

                for a, b in zip(As, Bs):
                    Iota[frame, jid, a, b] = solver.BoolVar('j[%i,%i,%i,%i]' % (frame, jid, a, b))
                    E_j.append((jid, a, b))

                s = solver.Sum(
                    Iota[frame, jid, int(a), int(b)] * func2(d) for a, b, d in distance)
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
                    Lambda[frame, jid1, jid2, a, b] = solver.BoolVar(
                        'l[%i,%i,%i,%i]' % (jid1, jid2, a, b))
                    E_l.append((jid1, jid2, a, b))

                W = limbs3d[lid]
                Scores = W[As, Bs]

                s = solver.Sum(
                    Get_Lambda(frame, jid1, jid2, a, b) * func3(s) for a, b, s in \
                    zip(As, Bs, Scores))
                Sum.append(s)

            # ===========================================
            # ONLY CONSIDER VALID EDGES
            # ===========================================
            for jid1, jid2, a, b in E_l:
                solver.Add(
                    Lambda[frame, jid1, jid2, a, b] * 2 <=
                    Nu[frame, jid1, a] + Nu[frame, jid2, b])

            for jid, a, b in E_j:
                solver.Add(
                    Iota[frame, jid, a, b] * 2 <=
                    Nu[frame, jid, a] + Nu[frame, jid, b])

            # ===========================================
            # HANDLE SYMMETRY CONSTRAINTS
            # ===========================================
            for jid1, jid2 in symmetric_joints:
                assert jid1 != jid2
                ABdistance = gm.calculate_distance_all4all(
                    points3d[jid1], points3d[jid2],
                    max_distance=min_symmetric_distance,
                    min_distance=0,
                    AB_are_the_same=False)
                As = ABdistance[:, 0].astype('int32')
                Bs = ABdistance[:, 1].astype('int32')
                for a, b in zip(As, Bs):
                    solver.Add(Nu[frame, jid1, a] + Nu[frame, jid2, b] <= 1)

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

            # ===========================================
            # HANDLE TRANSITIVITY CONSTRAINTS (2)
            # ===========================================
            for jid, a, b, c in Intra:
                assert a < b < c
                solver.Add(Get_Iota(frame, jid, a, b) +
                           Get_Iota(frame, jid, a, c) - 1 <=
                           Get_Iota(frame, jid, b, c))
                solver.Add(Get_Iota(frame, jid, a, b) +
                           Get_Iota(frame, jid, b, c) - 1 <=
                           Get_Iota(frame, jid, a, c))
                solver.Add(Get_Iota(frame, jid, a, c) +
                           Get_Iota(frame, jid, b, c) - 1 <=
                           Get_Iota(frame, jid, a, b))

            for jid1, a, b, jid2, c in Inter:
                solver.Add(
                    Get_Lambda(frame, jid1, jid2, a, c) +
                    Get_Lambda(frame, jid1, jid2, b, c) - 1 <=
                    Get_Iota(frame, jid1, a, b))
                solver.Add(
                    Get_Iota(frame, jid1, a, b) +
                    Get_Lambda(frame, jid1, jid2, a, c) - 1 <=
                    Get_Lambda(frame, jid1, jid2, b, c))
                solver.Add(
                    Get_Iota(frame, jid1, a, b) +
                    Get_Lambda(frame, jid1, jid2, b, c) - 1 <=
                    Get_Lambda(frame, jid1, jid2, a, c))

            # ===========================================
            # HANDLE CHOICE CONSTRAINTS
            # ===========================================
            for jid, a, b, c in Intra_choice:  # either take { ab OR ac }
                solver.Add(
                    Get_Iota(frame, jid, a, b) +
                    Get_Iota(frame, jid, a, c) <= 1
                )

            for jid1, a, jid2, b, jid3, c in Inter_choice:  # { ab OR ac }
                if jid1 == jid2:
                    assert jid3 != jid1
                    # if  [a]---[b]
                    #     |
                    #    (c)
                    solver.Add(
                        Get_Iota(frame, jid1, a, b) +
                        Get_Lambda(frame, jid1, jid3, a, c) <= 1
                    )
                elif jid2 == jid3:
                    # if  [a]
                    #     |   \
                    #    (b)   (c)
                    solver.Add(
                        Get_Lambda(frame, jid1, jid2, a, b) +
                        Get_Lambda(frame, jid1, jid3, a, c) <= 1
                    )
                elif jid1 == jid3:
                    # if  [a]---[c]
                    #     |
                    #    (b)
                    solver.Add(
                        Get_Lambda(frame, jid1, jid2, a, b) +
                        Get_Iota(frame, jid1, a, c) <= 1
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
            #print("\tresult:", RESULT)
            print('\n\tTotal cost:', solver.Objective().Value())