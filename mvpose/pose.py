"""
    Extract human poses from pafs and heatmaps
"""
from mvpose.algorithm.peaks2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.algorithm.limbs3d import Limbs3d
from mvpose.algorithm.graphcut import Graphcut, get_parameters
from mvpose.algorithm.candidate_selection import CandidateSelector
from mvpose.data.default_limbs import DEFAULT_LIMB_SEQ, \
    DEFAULT_MAP_IDX, DEFAULT_SENSIBLE_LIMB_LENGTH
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs,
             limbSeq=DEFAULT_LIMB_SEQ,
             limbMapIdx=DEFAULT_MAP_IDX,
             graphcut_params=None,
             sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
             debug=False, max_epi_distance=10):
    """

    :param Calib:
    :param heatmaps:
    :param pafs:
    :param limbSeq:
    :param limbMapIdx:
    :param graphcut_params: parameters for the graphcut
    :param sensible_limb_length:
    :param debug:
    :param max_epi_distance:
    :return:
    """
    n_cameras, h, w, n_limbs = pafs.shape
    n_limbs = int(n_limbs / 2)
    assert n_limbs == len(limbSeq)
    assert n_cameras == len(Calib)
    assert n_cameras == len(heatmaps)
    assert h == heatmaps.shape[1]
    assert w == heatmaps.shape[2]
    assert n_cameras > 2, 'The algorithm expects at least 3 views'

    # -------- step 1 --------
    # calculate 2d candidates
    # ------------------------
    _start = time()
    cand2d = Candidates2D(heatmaps, Calib)
    _end = time()
    if debug:
        print('step 1: elapsed', _end - _start)

    # -------- step 2 --------
    # triangulate 2d candidates
    # ------------------------
    _start = time()
    triangulation = Triangulation(cand2d, Calib, max_epi_distance)
    _end = time()
    if debug:
        print('step 2: elapsed', _end - _start)

    # -------- step 3 --------
    # calculate 3d limb weights
    # ------------------------
    _start = time()
    limbs3d = Limbs3d(triangulation.peaks3d_weighted,
                      Calib, pafs,
                      limbSeq, sensible_limb_length,
                      limbMapIdx)
    _end = time()
    if debug:
        print('step 3: elapsed', _end - _start)

    # -------- step 4 --------
    # solve optimization problem
    # ------------------------
    _start = time()
    if graphcut_params is None:
        graphcut_params = get_parameters()  # gets default params
    graphcut = Graphcut(graphcut_params,
                        triangulation.peaks3d_weighted,
                        limbs3d,
                        limbSeq, sensible_limb_length,
                        debug=debug
                        )
    _end = time()
    if debug:
        print('step 4: elapsed', _end - _start)

    # -------- step 5 --------
    # candidate selection
    # ------------------------
    _start = time()
    candSelector = CandidateSelector(
        graphcut.person_candidates, heatmaps,
        Calib, graphcut_params.min_nbr_joints)
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
