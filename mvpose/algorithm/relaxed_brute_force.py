from mvpose.algorithm.peaks2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.algorithm.limbs3d import Limbs3d
from mvpose.algorithm.meanshift import Meanshift
from mvpose.algorithm.candidate_selection import CandidateSelector
from mvpose.algorithm.graphcut import Graphcut
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs, settings,
             radius, sigma, max_iterations, between_distance,
             debug, silent=False):
    """
        use graph partitioning but simplify graph using meanshift
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param radius: radius for meanshift density estimation
    :param sigma: width of the gaussian in the meanshift
    :param max_iterations: cut-of threshold for meanshift
    :param between_distance: maximal distance between two points of a cluster
    :param debug:
    :param silent: allow debugging but do not print
    :return:
    """
    # -------- step 1 --------
    # calculate 2d candidates
    # ------------------------
    _start = time()
    cand2d = Candidates2D(heatmaps, Calib,
                          threshold=settings.hm_detection_threshold)
    _end = time()
    if debug and not silent:
        print('step 1: elapsed', _end - _start)

    # -------- step 2 --------
    # triangulate 2d candidates
    # ------------------------
    _start = time()
    triangulation = Triangulation(cand2d, Calib, settings.max_epi_distance)
    _end = time()
    if debug and not silent:
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
    eps = 0.1 / settings.scale_to_mm
    meanshift = Meanshift(triangulation.peaks3d_weighted,
                          float(radius), float(sigma), max_iterations, eps,
                          between_distance)
    _end = time()
    if debug and not silent:
        print('step 3: elapsed', _end - _start)

    # -------- step 4 --------
    # calculate 3d limb weights
    # ------------------------
    _start = time()
    limbs3d = Limbs3d(meanshift.centers3d,
                      Calib, pafs,
                      settings.limb_seq,
                      settings.sensible_limb_length,
                      settings.limb_map_idx,
                      oor_marker=0)
    _end = time()
    if debug and not silent:
        print('step 4: elapsed', _end - _start)

    # -------- step 5 --------
    # solve optimization problem
    # ------------------------
    _start = time()
    graphcut = Graphcut(settings,
                        meanshift.centers3d,
                        limbs3d,
                        debug=debug, silent=silent)
    _end = time()
    if debug and not silent:
        print('step 5: elapsed', _end - _start)

    # -------- step 7 --------
    # candidate selection  "filter out bad detections"
    # ------------------------
    _start = time()
    human_candidates = graphcut.person_candidates
    candSelector = CandidateSelector(
        human_candidates, heatmaps,
        Calib,
        settings.min_nbr_joints,
        hm_detection_threshold=settings.hm_detection_threshold,
        threshold_close_pair=settings.threshold_close_pair)
    _end = time()
    if debug and not silent:
        print('step 7: elapsed', _end - _start)

    # ------------------------
    # finalize
    # ------------------------
    if debug:
        Debug = namedtuple('Debug', [
            'candidates2d',
            'triangulation',
            'meanshift',
            'limbs3d',
            'graphcut',
            'human_candidates'
        ])
        Debug.candidates2d = cand2d
        Debug.triangulation = triangulation
        Debug.meanshift = meanshift
        Debug.limbs3d = limbs3d
        Debug.graphcut = graphcut
        Debug.human_candidates = human_candidates
        return Debug, candSelector.persons
    else:
        return candSelector.persons
