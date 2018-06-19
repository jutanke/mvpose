"""
    Extract human poses from pafs and heatmaps
"""
from mvpose.algorithm.settings import get_settings
from mvpose.algorithm import brute_force
from mvpose.algorithm import meanshift
from mvpose.algorithm import relaxed_brute_force


def validate_input(Calib, heatmaps, pafs, settings):
    """
        validates the input data
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :return:
    """
    if settings is None:
        settings = get_settings()  # get default parameters
    max_epi_distance = settings.max_epi_distance
    assert max_epi_distance > 0
    n_cameras, h, w, n_limbs = pafs.shape
    n_limbs = int(n_limbs / 2)
    limbSeq = settings.limb_seq
    assert n_limbs == len(limbSeq)
    assert n_cameras == len(Calib)
    assert n_cameras == len(heatmaps)
    assert h == heatmaps.shape[1]
    assert w == heatmaps.shape[2]
    assert n_cameras > 2, 'The algorithm expects at least 3 views'
    return settings


def estimate_heuristic(Calib, heatmaps, pafs, settings=None,
                       debug=False, use_greedy=False):
    """
        Brute-Force graph partitioning algorithm (np-hard)
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param use_greedy: {boolean} if True use the greedy algorithm, o/w graphcut
    :param debug:
    :return:
    """
    settings = validate_input(Calib, heatmaps, pafs, settings)
    radius = settings.ms_radius
    sigma = settings.ms_sigma
    max_iterations = settings.ms_max_iterations
    between_distance = settings.ms_between_distance
    if use_greedy:
        return meanshift.estimate(Calib, heatmaps, pafs, settings,
                                  radius, sigma, max_iterations,
                                  between_distance, debug)
    else:
        return relaxed_brute_force.estimate(Calib, heatmaps, pafs, settings,
                                            radius, sigma, max_iterations,
                                            between_distance,
                                            debug)


def estimate(Calib, heatmaps, pafs, settings=None, debug=False):
    """
        Brute-Force graph partitioning algorithm (np-hard)
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param debug:
    :return:
    """
    settings = validate_input(Calib, heatmaps, pafs, settings)
    return brute_force.estimate(Calib, heatmaps, pafs, settings, debug)
