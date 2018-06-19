from mvpose.pose import validate_input
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execute frame-wise detection
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("calib", len(Calib))
    print("hm", len(Heatmaps))
    print("paf", len(Pafs))
    _Limbs3d = []
    _Points3d = []
    for frame, (calib, heatmaps, pafs) in enumerate(zip(Calib, Heatmaps, Pafs)):
        if debug:
            print("handling frame ", frame)

        # -------- step 1 --------
        # calculate 2d candidates
        # ------------------------
        _start = time()
        cand2d = Candidates2D(heatmaps, calib,
                              threshold=settings.hm_detection_threshold)
        _end = time()
        if debug:
            print('\tstep 1: elapsed', _end - _start)
            Debug.candidates.append(cand2d)

        # -------- step 2 --------
        # triangulate 2d candidates
        # ------------------------
        _start = time()
        triangulation = Triangulation(cand2d, calib, settings.max_epi_distance)
        _end = time()
        if debug:
            print('\tstep 2: elapsed', _end - _start)
            Debug.triangulations.append(triangulation)

        # -------- step 3 --------
        # meanshift
        # ------------------------
        _start = time()
        radius = settings.ms_radius
        sigma = settings.ms_sigma
        max_iterations = settings.ms_max_iterations
        between_distance = settings.ms_between_distance
        eps = 0.1 / settings.scale_to_mm
        meanshift = Meanshift(triangulation.peaks3d_weighted,
                              float(radius), float(sigma), max_iterations, eps,
                              between_distance)
        _end = time()
        if debug:
            print('\tstep 3: elapsed', _end - _start)
            Debug.meanshifts.append(meanshift)

        # -------- step 4 --------
        # calculate 3d limb weights
        # ------------------------
        _start = time()
        limbs3d = Limbs3d(meanshift.centers3d,
                          calib, pafs,
                          settings.limb_seq,
                          settings.sensible_limb_length,
                          settings.limb_map_idx,
                          oor_marker=0)
        _end = time()
        if debug:
            print('\tstep 4: elapsed', _end - _start)

        _Points3d.append(meanshift.centers3d)
        _Limbs3d.append(limbs3d)

    # -------- step 4 --------
    # solve optimization problem
    # ------------------------
    _start = time()
    graphcut = GraphcutTracking(settings, _Points3d, _Limbs3d, debug=debug)
    _end = time()
    if debug:
        print('step 4: elapsed', _end - _start)

    if debug:
        return Debug