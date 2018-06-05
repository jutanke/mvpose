from mvpose.algorithm.peaks2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.algorithm.limbs3d import Limbs3d
from mvpose.algorithm.graphcut import Graphcut
from mvpose.algorithm.candidate_selection import CandidateSelector
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs, settings, debug=False):
    """
        Brute-Force graph partitioning algorithm (np-hard)
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param debug:
    :return:
    """
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

    # -------- step 3 --------
    # calculate 3d limb weights
    # ------------------------
    _start = time()
    limbs3d = Limbs3d(triangulation.peaks3d_weighted,
                      Calib, pafs,
                      settings.limb_seq,
                      settings.sensible_limb_length,
                      settings.limb_map_idx)
    _end = time()
    if debug:
        print('step 3: elapsed', _end - _start)

    # -------- step 4 --------
    # solve optimization problem
    # ------------------------
    _start = time()
    graphcut = Graphcut(settings,
                        triangulation.peaks3d_weighted,
                        limbs3d,
                        debug=debug)
    _end = time()
    if debug:
        print('step 4: elapsed', _end - _start)

    # -------- step 5 --------
    # candidate selection  "filter out bad detections"
    # ------------------------
    _start = time()
    candSelector = CandidateSelector(
        graphcut.person_candidates, heatmaps,
        Calib, settings.min_nbr_joints)
    _end = time()
    if debug:
        print('step 5: elapsed', _end - _start)

    if debug:
        Debug = namedtuple('Debug', [
            'candidates2d',
            'triangulation',
            'limbs3d',
            'graphcut'
        ])
        Debug.candidates2d = cand2d
        Debug.triangulation = triangulation
        Debug.limbs3d = limbs3d
        Debug.graphcut = graphcut

        return Debug, candSelector.persons
    else:
        return candSelector.persons