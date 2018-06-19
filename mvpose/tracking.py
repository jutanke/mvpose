from mvpose.pose import validate_input
from mvpose.algorithm.relaxed_brute_force import estimate
from mvpose.algorithm.peaks2d import Candidates2D
from mvpose.algorithm.triangulation import Triangulation
from mvpose.algorithm.meanshift import Meanshift
from mvpose.algorithm.limbs3d import Limbs3d
from mvpose.algorithm.brute_force_tracking import GraphcutTracking
from time import time
from collections import namedtuple


def track(Calib, Heatmaps, Pafs, settings=None, debug=False):
    """

    :param Calib: [ [ {mvpose.geometry.camera}, .. ] * n_cameras ] * n_frames
    :param Heatmaps: [ [n x h x w x j], ... ] * n_frames
    :param pafs: [ [n x h x w x 2*l], ... ] * n_frames
    :param settings: parameters for system
    :param debug: {boolean} if True print debug messages
    :return:
    """
    n_frames = len(Heatmaps)
    assert n_frames > 1
    assert len(Pafs) == n_frames

    # fix Calib: in some environments the cameras are fixed and do not
    # change: we then only get a single list of {mvpose.geometry.camera}'s
    # but in other instances we might have dynamic cameras where we have
    # a different calibration for each camera and frame. To be able to
    # handle both cases we 'equalize' them here by simply repeating the
    # same cameras if applicable
    if len(Calib) == 1:
        Calib_ = []
        for _ in range(n_frames):
            Calib_.append(Calib[0])
        Calib = Calib_

    settings = validate_input(Calib[0], Heatmaps[0], Pafs[0], settings)

    if debug:
        Debug = namedtuple('Debug', [
            'candidates',
            'triangulations',
            'meanshifts',
            'n_frames'
        ])
        Debug.candidates = []
        Debug.triangulations = []
        Debug.meanshifts = []
        Debug.n_frames = n_frames
        Debug.n_joints = Heatmaps.shape[4]
        Debug.humans = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execute frame-wise detection
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    radius = settings.ms_radius
    sigma = settings.ms_sigma
    max_iterations = settings.ms_max_iterations
    between_distance = settings.ms_between_distance
    humans = []
    for frame, (calib, heatmaps, pafs) in enumerate(zip(Calib, Heatmaps, Pafs)):
        _start = time()
        if debug:
            Debug_, candidates = estimate(calib, heatmaps, pafs, settings,
                                         radius=radius, sigma=sigma,
                                         between_distance=between_distance, silent=True,
                                         max_iterations=max_iterations, debug=debug)
            Debug.candidates.append(Debug_.candidates2d)
            Debug.triangulations.append(Debug_.triangulation)
            Debug.meanshifts.append(Debug_.meanshift)
            Debug.humans.append(candidates)
        else:
            candidates = estimate(calib, heatmaps, pafs, settings,
                                  radius=radius, sigma=sigma,
                                  between_distance=between_distance,
                                  max_iterations=max_iterations)

        humans.append(candidates)
        _end = time()
        if debug:
            print("handling frame ", frame)
            print('\telapsed', _end - _start)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # fit the candidates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for t1, t2 in zip(range(0, n_frames -1), range(1, n_frames)):
        pass

    if debug:
        return Debug

