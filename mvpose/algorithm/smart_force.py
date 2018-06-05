from mvpose.algorithm.peaks2d import Candidates2D
from collections import namedtuple
from time import time


def estimate(Calib, heatmaps, pafs, settings, close_points=10, debug=False):
    """
        Smarter brute-force approach (still np-hard)
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param close_points: distance in pixel in which two points in 2d
        are considered to be close-by to warrant a graph connection
    :param debug:
    :return:
    """
    # -------- step 1 --------
    # calculate 2d candidates
    # ------------------------
    _start = time()
    cand2d = Candidates2D(heatmaps, Calib,
                          threshold=settings.hm_detection_threshold)

    # -------- step 2 --------
    # graph setup
    # ------------------------
    graph = SmartGraph(Calib, cand2d, settings)

    # ------------------------
    # finish up
    # ------------------------
    _end = time()
    if debug:
        print('step 1: elapsed', _end - _start)

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


class SmartGraph:
    """

    """

    def __index__(self, Calib, candidates2d, settings):
        peaks2d = candidates2d.peaks2d_undistorted
        n_cameras = len(peaks2d)
        n_joints = candidates2d.n_joints
        assert n_cameras == len(Calib)



