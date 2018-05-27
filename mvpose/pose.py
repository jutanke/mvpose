"""
    Extract human poses from pafs and heatmaps
"""
from mvpose.algorithm.peaks2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.data.default_limbs import DEFAULT_LIMB_SEQ, \
    DEFAULT_MAP_IDX, DEFAULT_SENSIBLE_LIMB_LENGTH
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs,
             limbSeq=DEFAULT_LIMB_SEQ,
             limbMapIdx=DEFAULT_MAP_IDX,
             sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
             debug=False, max_epi_distance=10):
    """

    :param Calib:
    :param heatmaps:
    :param pafs:
    :param limbSeq:
    :param limbMapIdx:
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

    _start = time()
    cand2d = Candidates2D(heatmaps, Calib)
    _end = time()
    if debug:
        print('step 1: elapsed', _end - _start)

    _start = time()
    triangulation = Triangulation(cand2d, max_epi_distance)
    _end = time()
    if debug:
        print('step 2: elapsed', _end - _start)

    if debug:
        Debug = namedtuple('Debug', [
            'candidates2d',
            'triangulation'])
        Debug.candidates2d = cand2d
        Debug.triangulation = triangulation

        return Debug