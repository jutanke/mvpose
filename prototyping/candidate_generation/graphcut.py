import json
Settings = json.load(open('../../settings.txt'))
print('\n')
# ------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile
import sys
from time import time
import cv2
import numpy as np
from os.path import join, isdir
sys.path.insert(0,'../../../easy_multi_person_pose_estimation')
sys.path.insert(0,'../../')
import mvpose.data.transform as tfm
import mvpose.geometry.geometry as gm
from mvpose.geometry import stereo
import mvpose.pose_estimation.heatmaps as mvhm
from mvpose.candidates import peaks as mvpeaks
import mvpose.pose_estimation.part_affinity_fields as mvpafs
from mvpose.algorithm import graphcut
from cselect import color as cs
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH, DEFAULT_MAP_IDX
from mvpose.plot.limbs import draw_vector_field

root = join(Settings['data_root'], 'pak')

from poseestimation import model
pe = model.PoseEstimator()


from pak.datasets.UMPM import UMPM
user = Settings['UMPM']['username']
pwd = Settings['UMPM']['password']

X, Y, Calib = tfm.get_from_umpm(root, 'p2_free_1', user, pwd)

# interesting frames: [340, 215, 250]
FRAME = 0

Im = np.array([X[0][FRAME], X[1][FRAME], X[2][FRAME], X[3][FRAME]])
with_gpu = False

if with_gpu:
    _start = time()
    heatmaps, pafs = pe.predict_pafs_and_heatmaps(I)
    _end = time(); print('elapsed:', _end - _start)
else:
    hm_file = '/tmp/heatmaps' + str(FRAME) + '.npy'
    paf_file = '/tmp/pafs' + str(FRAME) + '.npy'

    if isfile(hm_file) and isfile(paf_file):
        heatmaps = np.load(hm_file)
        pafs = np.load(paf_file)
    else:
        heatmaps = []; pafs = []
        for im in Im:
            _start = time()
            hm, paf = pe.predict_pafs_and_heatmaps(im)
            heatmaps.append(np.squeeze(hm))
            pafs.append(np.squeeze(paf))
            _end = time()
            print('elapsed:', _end - _start)
        heatmaps = np.array(heatmaps)
        pafs = np.array(pafs)
        np.save(hm_file, heatmaps)
        np.save(paf_file, pafs)

import mvpose.plot.limbs as pltlimbs

colors = cs.lincolor(19)/255

r = 200
_start = time()
Gr = graphcut.GraphCutSolver(heatmaps, pafs, Calib, r, debug=True)
_end = time()
print('elapsed', _end - _start)

from ortools.linear_solver import pywraplp as mip
from mvpose.geometry import geometry as gm
from mvpose.candidates.transitivity import TransitivityLookup

DEBUG = []

def cluster(points3d, limbs3d, limbSeq, sensible_limb_length,
            radius=50, max_radius=300):
    assert len(limbs3d) == len(limbSeq)
    assert len(limbs3d) == len(sensible_limb_length)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # formulas
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pboost_big = lambda x: np.log((x+1) / (2 * (0.5 * (-x -1)+1)))
    pboost_small = lambda x: np.log(x/(1-x))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    solver = mip.Solver('m', mip.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    #solver = mip.Solver('m')

    D = []  # all nodes of the graph    (jid, a)
    E_j = []  # all edges on the joints (jid, a, b)
    E_l = []  # all edges on the limbs  (jid1, jid2, a, b)

    Nu = {}  # jid, a
    Iota = {}  # jid a b
    Lambda = {}  # jid1 jid2 a b

    Sum = []

    for jid, pts3d in enumerate(points3d):
        # =================
        # ~~~ handle NU ~~~
        # =================

        left = pts3d[:,3]
        right = pts3d[:,4]
        unary = np.multiply(left, right)
        n = len(unary)
        for idx in range(n):
            Nu[jid, idx] = solver.BoolVar('nu[%i,%i]' % (jid, idx))
            D.append((jid, idx))

        s = solver.Sum(
            Nu[jid, idx] * pboost_small(unary[idx]) for idx in range(n)
        )
        Sum.append(s)

        # ==========================
        # ~~~ handle intra-class ~~~
        # ==========================
        # (a, b, distance)
        distance = gm.calculate_distance_all4all(
            Points3d[jid], Points3d[jid], max_radius,
            AB_are_the_same=True)
        As = distance[:,0].astype('int32')
        Bs = distance[:,1].astype('int32')

        for a, b in zip(As, Bs):
            Iota[jid, a, b] = solver.BoolVar('j[%i,%i,%i]' % (jid, a, b))
            E_j.append((jid, a, b))

        s = solver.Sum(
            Iota[jid, int(a), int(b)] * np.tanh((-d) + radius) for a,b,d in distance
        )
        Sum.append(s)

    # ==========================
    # ~~~ handle inter-class ~~~
    # ==========================
    for lid, ((jid1,jid2), (mindist, maxdist)) in \
        enumerate(zip(limbSeq, sensible_limb_length)):

        flip_jids = False
        if jid1 > jid2:
            # to drop duplicate edges we assume that jid1 < jid2
            # while calculating its neighbors
            temp = jid2; jid2 = jid1; jid1 = temp
            flip_jids = True

        # (a, b, distance)
        ABdistance = gm.calculate_distance_all4all(
            Points3d[jid1], Points3d[jid2], maxdist,
            min_distance=mindist,
            AB_are_the_same=False)
        As = ABdistance[:,0].astype('int32')
        Bs = ABdistance[:,1].astype('int32')
        distance = ABdistance[:,2]

        for a,b in zip(As, Bs):
            Lambda[jid1, jid2, a, b] = solver.BoolVar(
                'l[%i,%i,%i,%i]' % (jid1, jid2, a, b))
            E_l.append((jid1, jid2, a, b))

        W = limbs3d[lid]
        if flip_jids:
            Scores = W[Bs,As]
        else:
            Scores = W[As,Bs]

        s = solver.Sum(
            Lambda[jid1, jid2, a, b] * pboost_big(s) for a,b,s in\
            zip(As, Bs, Scores))
        Sum.append(s)

    # =========================
    # ~~~ handle conditions ~~~
    # =========================
    for jid1, jid2, a, b in E_l:
        solver.Add(
            Lambda[jid1, jid2, a, b] * 2 <= Nu[jid1, a] + Nu[jid2, b])

    for jid, a, b in E_j:
        solver.Add(
            Iota[jid, a, b] * 2 <= Nu[jid, a] + Nu[jid, b])


    transitivity_lookup = TransitivityLookup(D, E_l, E_j)

    Intra = []  #  [ (jid, a, b, c), ...]
    Inter = []  # [ (jid1, a, b, jid2, c), ...]

    for q in D:
        intra, inter = transitivity_lookup.query(*q)
        Intra += intra
        Inter += inter


    DEBUG.append(Inter)
    DEBUG.append(Lambda)


    for jid, a, b, c in Intra:
        solver.Add(
            Iota[jid, a, b] + Iota[jid, a, c] - 1 <= Iota[jid, b, c])
        solver.Add(
            Iota[jid, a, c] + Iota[jid, b, c] - 1 <= Iota[jid, a, b])
        solver.Add(
            Iota[jid, a, b] + Iota[jid, a, c] - 1 <= Iota[jid, b, c])





#     print(Lambda.keys())

    for jid1, a, b, jid2, c in Inter:
        solver.Add(
            Lambda[jid1, jid2, a, c] + Iota[jid1, a, b] - 1 <=\
            Lambda[jid1, jid2, b, c])
        solver.Add(
            Lambda[jid1, jid2, b, c] + Iota[jid1, a, b] - 1 <= Lambda[jid1, jid2, a, c])
        solver.Add(
            Lambda[jid1, jid2, a, c] + Lambda[jid1, jid2, b, c] - 1 <= Iota[jid1, a, b])


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("D", len(D))
    print("E_j", len(E_j))
    print("E_l", len(E_l))
    print("Sums:", len(Sum))
    print("intra:", len(Intra))
    print("inter:", len(Inter))

    # ~~~ execute optimization ~~~
    solver.Maximize(sum(Sum))
    solver.Solve()
    print("Time = ", solver.WallTime(), " ms")


    return Nu, Iota, Lambda

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Points3d = Gr.peaks3d_weighted
Limbs3d = Gr.limbs3d

_start = time()
Nu, Iota, Lambda = cluster(Points3d, Limbs3d, DEFAULT_LIMB_SEQ, DEFAULT_SENSIBLE_LIMB_LENGTH)
_end = time()
print('elapsed', _end - _start)

QQ = []

for jid, pts3d in enumerate(Points3d):
    for i, p in enumerate(pts3d):
        qq = Nu[jid, i]
        QQ.append(qq)

print("www")

print('qqq')
print("lambda:", Lambda[(5, 17, 6, 1)])

#print(QQ[0])
