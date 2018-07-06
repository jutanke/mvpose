from mvpose.pose import validate_input
from mvpose.algorithm.relaxed_brute_force import estimate
from mvpose.algorithm.track_graph_partitioning import GraphPartitioningTracker
from mvpose.algorithm.settings import get_tracking_settings
from time import time
from collections import namedtuple
import numpy as np
import networkx as nx


def track(Calib, Imgs, Heatmaps, Pafs,
          settings=None, tracking_setting=None, debug=False):
    """

    :param Calib: [ [ {mvpose.geometry.camera}, .. ] * n_cameras ] * n_frames
    :param Imgs: [ [n x h x w x 3], .... ] * n_frames
    :param Heatmaps: [ [n x h x w x j], ... ] * n_frames
    :param pafs: [ [n x h x w x 2*l], ... ] * n_frames
    :param settings: parameters for system
    :param tracking_setting: {mvpose.algorithm.settings.Tracking_Settings}
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
    if tracking_setting is None:
        tracking_setting = get_tracking_settings()  # default params

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
    _start = time()
    graph_part = GraphPartitioningTracker(Calib, Imgs, humans, debug,
                                          tracking_setting)
    _end = time()
    if debug:
        Debug.track_partitioning = graph_part
        print('graph partitioning: elapsed', _end - _start)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create tracks
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _start = time()
    tracks = []
    for frame, hm in enumerate(humans):
        tracks.append([-1 for _ in range(len(hm))])

    for global_pid, comp in enumerate(nx.connected_components(graph_part.G)):
        for nid in comp:
            node = graph_part.G.nodes[nid]
            t, local_pid = node['key']
            tracks[t][local_pid] = global_pid

    _end = time()

    if debug:
        print('parse partitioning', _end - _start)
        return Debug, np.array(tracks), humans
    else:
        return np.array(tracks), humans

