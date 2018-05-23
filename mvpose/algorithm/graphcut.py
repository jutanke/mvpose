from mvpose.pose_estimation import limb_weights
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH, DEFAULT_MAP_IDX
from mvpose.algorithm.meanshift import find_all_modes
from mvpose.pose_estimation import part_affinity_fields as mvpafs
import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment
import mvpose.pose_estimation.heatmaps as mvhm
from mvpose.candidates import peaks as mvpeaks
from scipy.special import comb
import cv2
from time import time
from ortools.linear_solver import pywraplp as mip
from mvpose.candidates.transitivity import TransitivityLookup
import networkx as nx


def merge(person):
    result = [None] * len(person)
    for jid, points3d in enumerate(person):
        if points3d is not None:
            Pos = points3d[:, 0:3]
            W = np.expand_dims(points3d[:, 3] * points3d[:, 4], axis=1)
            S = np.sum(W)
            WPos = np.multiply(Pos, W)
            WPos = np.sum(WPos, axis=0) / S
            result[jid] = WPos
    return result


def project_human_to_2d(human3d, cam):
    human2d = [None] * len(human3d)
    for jid, pt3d in enumerate(human3d):
        if pt3d is not None:
            Pt = np.array([pt3d])
            K, rvec, tvec, _ = gm.get_camera_parameters(cam)
            points2d = np.squeeze(cv2.projectPoints(Pt, rvec, tvec, K, 0)[0])
            human2d[jid] = points2d
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


class GraphCutSolver:

    def __init__(self, Heatmaps, Pafs, Calib, r, sigma=-1,
                 limbSeq=DEFAULT_LIMB_SEQ,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
                 limbMapIdx=DEFAULT_MAP_IDX, debug=False,
                 max_epi_distance=10,
                 ):
        """
            Extract 3d pose from images and cameras
        :param Heatmaps: list of heatmaps
        :param Pafs: list of part affinity fields
        :param Calib: list of calibrations per camera
        :param r:
        :param sigma:
        :param limbSeq:
        :param sensible_limb_length:
        """
        n_cameras, h, w, n_limbs = Pafs.shape
        n_limbs = int(n_limbs/2)
        assert r > 0
        assert n_limbs == len(DEFAULT_LIMB_SEQ)
        assert n_cameras == len(Calib)
        assert n_cameras == len(Heatmaps)
        assert h == Heatmaps.shape[1]
        assert w == Heatmaps.shape[2]
        assert n_cameras >= 3, 'The algorithm expects at least 3 views'
        if sigma == -1:
            sigma = r

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 1: get all peaks
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _start = time()
        self.peaks2d = []
        self.peaks2d_undistorted = []

        self.undistort_maps = []
        self.Calib_undistorted = []

        n_joints = -1

        for cid, cam in enumerate(Calib):
            hm = Heatmaps[cid]
            peaks = mvhm.get_all_peaks(hm)
            if n_joints < 0:
                n_joints = peaks.n_joints
            else:
                assert n_joints == peaks.n_joints
            self.peaks2d.append(peaks)

            # -- undistort peaks --
            K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)
            hm_ud, K_new = gm.remove_distortion(hm, cam)
            h,w,_ = hm.shape

            mapx, mapy = \
                cv2.initUndistortRectifyMap(
                K, distCoef, None, K_new, (w, h), 5)
            self.undistort_maps.append((mapx, mapy))

            peaks_undist = mvpeaks.Peaks.undistort(peaks, mapx, mapy)
            self.peaks2d_undistorted.append(peaks_undist)

            self.Calib_undistorted.append({
                'K': K_new,
                'distCoeff': 0,
                'rvec': rvec,
                'tvec': tvec
            })
        _end = time()
        if debug:
            print("[GRAPHCUT] step1 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 2: triangulate all points
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _start = time()
        Peaks3d = [np.zeros((0, 7))] * n_joints

        for cam1 in range(n_cameras - 1):
            K1, rvec1, tvec1, distCoef1 = \
                gm.get_camera_parameters(self.Calib_undistorted[cam1])
            assert distCoef1 == 0
            peaks1 = self.peaks2d_undistorted[cam1]

            for cam2 in range(cam1 + 1, n_cameras):
                K2, rvec2, tvec2, distCoef2 = \
                    gm.get_camera_parameters(self.Calib_undistorted[cam2])
                assert distCoef2 == 0
                peaks2 = self.peaks2d_undistorted[cam2]

                peaks3d = stereo.triangulate_with_weights(
                    peaks1, K1, rvec1, tvec1,
                    peaks2, K2, rvec2, tvec2, max_epi_distance=max_epi_distance
                )
                assert len(peaks3d) == n_joints

                for k in range(n_joints):
                    Peaks3d[k] = np.concatenate(
                        [Peaks3d[k], peaks3d[k]], axis=0
                    )

        self.peaks3d_weighted = Peaks3d
        _end = time()
        if debug:
            print("[GRAPHCUT] step2 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 3: reproject all 3d points onto all 2d views
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # IMPORTANT: the points have to be distorted so that
        #            we can safely work with the pafs (which
        #            are only defined in the distorted world)

        # defined as follows:
        #       (x, y, score1, score2, point-line-dist1, point-line-dist1)

        _start = time()
        # TODO: this is bug-heavy memorywise...
        self.candidates2d_undistorted = [[np.zeros((0, 6))] * n_joints] * n_cameras
        self.candidates2d = [[np.zeros((0, 6))] * n_joints] * n_cameras

        for cid, cam in enumerate(self.Calib_undistorted):
            K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)
            assert distCoef == 0

            Cand2d = self.candidates2d_undistorted[cid].copy()
            Cand2d_dist = self.candidates2d[cid].copy()
            assert len(Cand2d) == n_joints
            assert len(Cand2d_dist) == n_joints

            mapx, mapy = self.undistort_maps[cid]

            for k in range(n_joints):
                Pts3d = Peaks3d[k][:, 0:3]
                n_points = Pts3d.shape[0]
                if n_points > 0:

                    pts2d, mask = gm.reproject_points_to_2d(Pts3d, rvec, tvec, K, w, h)

                    W = np.squeeze(Peaks3d[k][mask, 3:].copy())
                    if len(W.shape) == 1:
                        W = np.expand_dims(W, axis=0)

                    pts2d = pts2d[mask]
                    qq = np.concatenate([pts2d, W], axis=1)
                    Cand2d[k] = qq
                    assert Cand2d[k].shape[1] == 6

                    pts2d_distorted = Cand2d[k].copy()
                    dist_xy = gm.distort_points(pts2d[:, 0:2], mapx, mapy)

                    pts2d_distorted[:, 0] = dist_xy[:, 0]
                    pts2d_distorted[:, 1] = dist_xy[:, 1]

                    Cand2d_dist[k] = pts2d_distorted

            self.candidates2d_undistorted[cid] = Cand2d
            self.candidates2d[cid] = Cand2d_dist
        _end = time()
        if debug:
            print("[GRAPHCUT] step3 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 4: calculate the weights for the limbs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _start = time()
        CAMERA_NORM = comb(n_cameras, 2)  # this is needed to make the 3d pafs be between -1 .. 1
        self.limbs3d = [None] * n_limbs

        assert len(limbSeq) == len(sensible_limb_length)
        for idx,((a,b), (length_min, length_max), (pafA, pafB)) in \
                enumerate(zip(limbSeq, sensible_limb_length, limbMapIdx)):
            # 3d peaks are setup as follows:
            #       (x,y,z,score1,score2,p2l-dist1,p2l-dist2)
            candA3d = Peaks3d[a]
            candB3d = Peaks3d[b]

            nA = len(candA3d)
            nB = len(candB3d)

            W = np.zeros((nA, nB))

            if nA > 0 and nB > 0:
                for cid, cam in enumerate(Calib):
                    K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)

                    U = Pafs[cid, :, :, pafA]
                    V = Pafs[cid, :, :, pafB]

                    ptsA2d, maskA = gm.reproject_points_to_2d(
                        candA3d[:, 0:3], rvec, tvec, K, w, h, distCoef=distCoef, binary_mask=True)
                    ptsB2d, maskB = gm.reproject_points_to_2d(
                        candB3d[:, 0:3], rvec, tvec, K, w, h, distCoef=distCoef, binary_mask=True)
                    maskA = maskA == 1
                    maskB = maskB == 1

                    ptA_candidates = []
                    ptB_candidates = []
                    pair_candidates = []

                    for i, (ptA, ptA3d, is_A_on_screen) in enumerate(zip(ptsA2d, candA3d, maskA)):
                        if not is_A_on_screen:
                            continue
                        for j, (ptB, ptB3d, is_B_on_screen) in enumerate(zip(ptsB2d, candB3d, maskB)):
                            if not is_B_on_screen:
                                continue
                            distance = la.norm(ptA3d[0:3] - ptB3d[0:3])
                            if length_min < distance < length_max:
                                ptA_candidates.append(ptA)
                                ptB_candidates.append(ptB)
                                pair_candidates.append((i, j))

                    if len(ptA_candidates) > 0:
                        line_int = mvpafs.calculate_line_integral_elementwise(
                            np.array(ptA_candidates), np.array(ptB_candidates), U, V)
                        assert len(line_int) == len(pair_candidates)
                        line_int = np.squeeze(line_int / CAMERA_NORM)
                        if len(line_int.shape) == 0:  # this happens when line_int.shape = (1, 1)
                            line_int = np.expand_dims(line_int, axis=0)
                        for score, (i, j) in zip(line_int, pair_candidates):
                            W[i, j] += score

            self.limbs3d[idx] = W
        _end = time()
        if debug:
            print("[GRAPHCUT] step4 elapsed:", _end - _start)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 5: create optimization problem
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # creating poses is done in stages: first we optimize the
        # right limb candidates
        _start = time()

        # ===========================================
        # PARAMETERS
        # ===========================================
        max_radius = 300
        radius = 50
        iota_scale = 1
        min_nbr_joints = 8

        # ===========================================
        # COST  FUNCTIONS
        # ===========================================
        pboost_big = lambda x: np.log((x + 1) / (2 * (0.5 * (-x - 1) + 1))) * 2
        pboost_small = lambda x: np.log(x / (1 - x))
        func1 = lambda u: np.tanh(pboost_small(u))
        func2 = lambda d: (-np.tanh((d - radius) / radius) * iota_scale)
        func3 = lambda x: pboost_big(x)

        # ===========================================
        # CREATE COST AND BOOLEAN VARIABLES
        # ===========================================
        solver = mip.Solver('m', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        points3d = self.peaks3d_weighted
        limbs3d = self.limbs3d

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
            left = pts3d[:, 3]
            right = pts3d[:, 4]
            unary = np.multiply(left, right)
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
                points3d[jid1], points3d[jid2], max_distance=maxdist,
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

        self.person_candidates = valid_persons

        # ===========================================
        # DEBUG: PRINT SOME STATISTICS
        # ===========================================
        _end = time()
        assert len(Nu) == len(D)
        assert len(E_j) == len(Iota)
        assert len(E_l) == len(Lambda)

        if debug:
            print('\n-------------------------------------------')
            print("\tNu:\t", len(Nu))
            print("\tIota:\t", len(Iota))
            print("\tLambda:\t", len(Lambda))
            print('-------------------------------------------\n')
            print('')
            print('#persons:', len(persons))
            print("[GRAPHCUT] step5 elapsed", _end - _start)

        Humans = [merge(p) for p in valid_persons]

        # --
        hm_detection_threshold = 0.1
        threshold_close_pair = 10  # in pixels

        n = len(Humans)
        n_cams = len(self.Calib_undistorted)

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
            for cid, cam in enumerate(self.Calib_undistorted):
                human2d_a = project_human_to_2d(human3d_a, cam)
                # ==============================================
                # check in how many views two persons co-incide
                # ==============================================
                for b in range(a + 1, n):
                    human3d_b = Humans[b]
                    human2d_b = project_human_to_2d(human3d_b, cam)

                    # check if co-incide
                    distance = calculate2d_proximity(human2d_a, human2d_b)
                    count_close_pairs = 0
                    for d in distance:
                        if d < 0:
                            continue
                        if d < threshold_close_pair:
                            count_close_pairs += 1

                    if count_close_pairs > min_nbr_joints:
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
                        x, y = np.around(pt2d).astype('int32')
                        if x > 0 and y > 0 and x < w and y < h:
                            score = hm[y, x, jid]
                            believe[jid] = score

                total = np.sum((np.array(believe) > hm_detection_threshold))
                if total > min_nbr_joints:
                    Visibility_Table[cid, a] = max(FLAG_VISIBLE, Visibility_Table[cid, a])
                else:
                    Visibility_Table[cid, a] = FLAG_NOT_VISIBLE

        Valid_Humans = []
        for human3d, visibility_in_cams in zip(Humans, np.transpose(Visibility_Table)):
            valid_cams = np.sum(visibility_in_cams == 1)
            if valid_cams > 1:  # valid in at least 2 views
                Valid_Humans.append(human3d)

        self.persons = Valid_Humans
