from mvpose.settings import get_settings
from mvpose.algorithm.candidates2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.algorithm.meanshift import Meanshift
from mvpose.algorithm.reweighting import ReWeighting
from mvpose.algorithm.limbs3d import Limbs3d
from mvpose.algorithm.graph_partitioning import GraphPartitioning
from mvpose.algorithm.candidate_selection import CandidateSelector
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs, settings=None, debug=False):
    """
    Estimate the human poses
    :param Calib:
    :param heatmaps:
    :param pafs:
    :param settings:
    :param debug: {Boolean} if true print debug information
    :return:
    """
    if settings is None:
        settings = get_settings()

    # -------- step 1 --------
    # calculate 2d candidates
    # ------------------------
    _start = time()
    cand2d = Candidates2D(heatmaps, Calib,
                          threshold=settings.hm_detection_threshold)
    _end = time()
    if debug:
        print('step 1: elapsed', _end - _start)

    # -------- step 2 --------
    # triangulate 2d candidates
    # ------------------------
    _start = time()
    triangulation = Triangulation(cand2d, Calib, settings.max_epi_distance)
    _end = time()
    if debug:
        print('step 2: elapsed', _end - _start)

    # check if there are NO detections
    valid_joints = 0
    for joint in triangulation.peaks3d_weighted:
        if len(joint) > 0:
            valid_joints += 1
    if valid_joints < settings.min_nbr_joints:
        if debug:
            print('\tnot enough 3d points found: early stopping!', valid_joints)
            Debug = namedtuple('Debug', [
                'candidates2d',
                'triangulation'])
            Debug.candidates2d = cand2d
            Debug.triangulation = triangulation
            return Debug, []
        else:
            return []

    # -------- step 3 --------
    # meanshift
    # ------------------------
    _start = time()
    radius = settings.ms_radius
    sigma = radius
    max_iterations = settings.ms_max_iterations
    between_distance = settings.ms_between_distance

    eps = 0.1 / settings.scale_to_mm
    meanshift = Meanshift(triangulation.peaks3d_weighted,
                          float(radius), float(sigma), max_iterations, eps,
                          between_distance, n_cameras=len(Calib))

    #reweighting = ReWeighting(Calib, heatmaps, meanshift.centers3d)
    points3d = meanshift.centers3d
    _end = time()
    if debug:
        print('step 3: elapsed', _end - _start)

    # -------- step 4 --------
    # calculate 3d limb weights
    # ------------------------
    _start = time()
    limbs3d = Limbs3d(points3d,
                      Calib, pafs,
                      settings.limb_seq,
                      settings.sensible_limb_length,
                      settings.limb_map_idx,
                      oor_marker=0)
    _end = time()
    if debug:
        print('step 4: elapsed', _end - _start)

    # -------- step 5 --------
    # solve optimization problem
    # ------------------------
    _start = time()
    graphcut = GraphPartitioning(points3d,
                        limbs3d, settings, debug=debug)
    _end = time()
    if debug:
        print('step 5: elapsed', _end - _start)

    # -------- step 6 --------
    # candidate selection  "filter out bad detections"
    # ------------------------
    _start = time()
    human_candidates = graphcut.person_candidates
    candSelector = CandidateSelector(
        human_candidates, heatmaps,
        Calib,
        settings.min_nbr_joints,
        conflict_covering=settings.pp_conflict_overlap,
        hm_detection_threshold=settings.hm_detection_threshold,
        debug=debug)
    _end = time()
    if debug:
        print('step 6: elapsed', _end - _start)

    # ------------------------
    # finalize
    # ------------------------
    if debug:
        Debug = namedtuple('Debug', [
            'candidates2d',
            'triangulation',
            'meanshift',
            'reweighting',
            'limbs3d',
            'graphcut',
            'human_candidates',
            'candidate_selector'
        ])
        Debug.candidates2d = cand2d
        Debug.triangulation = triangulation
        Debug.meanshift = meanshift
        #Debug.reweighting = reweighting
        Debug.limbs3d = limbs3d
        Debug.graphcut = graphcut
        Debug.candidate_selector = candSelector
        Debug.human_candidates = human_candidates
        return Debug, candSelector.persons
    else:
        return candSelector.persons, graphcut.objective_value, meanshift.centers3d, limbs3d.limbs3d, triangulation.peaks3d_weighted
