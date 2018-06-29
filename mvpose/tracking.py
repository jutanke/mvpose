from mvpose.pose import validate_input
from mvpose.algorithm.relaxed_brute_force import estimate
from mvpose.algorithm.track_graph_partitioning import GraphPartitioningTracker
from mvpose.algorithm.temporal import avg_distance
from time import time
from collections import namedtuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from reid import reid


def track(Calib, Imgs, Heatmaps, Pafs, settings=None, debug=False):
    """

    :param Calib: [ [ {mvpose.geometry.camera}, .. ] * n_cameras ] * n_frames
    :param Imgs: [ [n x h x w x 3], .... ] * n_frames
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
            'n_frames',
            'track_partitioning'
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
    tracks = []
    for frame, hm in enumerate(humans):
        if frame == 0:
            tracks.append(list(range(len(hm))))
        else:
            tracks.append([-1 for _ in range(len(hm))])

    graph_part = GraphPartitioningTracker(Calib, Imgs, humans, debug,
                                          settings.tr_valid_person_bb_area)
    if debug:
        Debug.track_partitioning = graph_part

    # current_pid = len(tracks[0])
    # max_distance = settings.track_max_distance
    # for t1, t2 in zip(range(0, n_frames - 1), range(1, n_frames)):
    #     pidsA = tracks[t1]
    #     pidsB = tracks[t2]
    #     candsA = humans[t1]
    #     candsB = humans[t2]
    #     nA = len(candsA)
    #     nB = len(candsB)
    #     if nA > 0 and nB > 0:
    #         D = np.zeros((nA, nB))
    #         for i, candA in enumerate(candsA):
    #             for j, candB in enumerate(candsB):
    #                 D[i, j] = avg_distance(candA, candB)
    #
    #         row_ind, col_ind = linear_sum_assignment(D)
    #         for a, b in zip(row_ind, col_ind):
    #             dist = D[a, b]
    #             if dist < max_distance:
    #                 pidsB[b] = pidsA[a]
    #             else:  # add a new person
    #                 pidsB[b] = current_pid
    #                 current_pid += 1

    if debug:
        return Debug, np.array(tracks), humans
    else:
        return np.array(tracks), humans

