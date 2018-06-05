from mvpose.algorithm.peaks2d import Candidates2D
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs, settings, debug):
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

    # ------------------------
    # finalize
    # ------------------------
    if debug:
        Debug = namedtuple('Debug', [
            'candidates2d',
            'triangulation',
            'limbs3d',
            'graphcut'
        ])
        Debug.candidates2d = cand2d

        return Debug, []
    else:
        return []