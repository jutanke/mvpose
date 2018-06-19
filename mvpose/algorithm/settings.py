from collections import namedtuple
from mvpose.data.default_limbs import DEFAULT_LIMB_SEQ, \
    DEFAULT_MAP_IDX, DEFAULT_SENSIBLE_LIMB_LENGTH, DEFAULT_SYMMETRIC_JOINTS


def get_settings(min_nbr_joints=7, gc_iota_scale=1,
                 gc_max_radius=300, gc_radius=50,
                 hm_detection_threshold=0.1,
                 threshold_close_pair=10, scale_to_mm=1,
                 max_epi_distance=10,
                 ms_radius=30, ms_sigma=None, ms_max_iterations=1000,
                 limb_seq=DEFAULT_LIMB_SEQ,
                 limb_map_idx=DEFAULT_MAP_IDX,
                 symmetric_joints=DEFAULT_SYMMETRIC_JOINTS,
                 min_symmetric_distance=50,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH):
    """
        gets the parameters that are needed for the the program
        !!Careful!!: to set the parameters that are defined in world coordinates
        the unit 'mm' MUST BE used. HowIT IS IN!ever, inside the setting object this
        units will be scaled according to {scale_to_mm}.

    :param min_nbr_joints: number of joints that are needed
        to be a 'valid' human pose detection
    :param gc_iota_scale: how to scale the cost function for iota
    :param gc_max_radius: maximal radius for internal distances
        of 3d joint candidates in mm
        (GraphCut)
    :param gc_radius: drop-off value after which internal weights
        between 3d joint candidates are negative in mm (GraphCut)
    :param hm_detection_threshold: confidence map threshold for
        when a peak is a detection and when it is an outlier [0 .. 1]
    :param threshold_close_pair: number of joints that need to be
        "close" for two detections to be considered a collision in
        the back-projection (in pixel)
    :param scale_to_mm: multiplier that scales the world coordinates
        into mm. E.g. if the world coordinates are in m, than
        scale_to_mm = 1000
    :param max_epi_distance: maximum distance to the respective
        epipolar line that allows two points to be triangulated
        (in pixel)
    :param ms_radius: range for the meanshift density estimation (in [mm])
    :param ms_sigma: width of the gaussian in the meanshift
    :param ms_max_iterations: cut-of threshold for meanshift
    :param limb_seq: [ (a, b), (a, c), ... ] list of limbs by joint connection
    :param limb_map_idx: maps the limb id to the part affinity field positions
    :param symmetric_joints: [ (a, b), ... ] list of joints that are symmetric (left/right arm)
    :param min_symmetric_distance: {float} minimal allowed distance between symmetric joints in [mm]
    :param sensible_limb_length: [ (min, max), (min, max), ... ] defines the
        sensible length in mm
    :return:
    """
    params = namedtuple('Settings', [
        'gc_max_radius',
        'gc_radius',
        'min_nbr_joints',
        'gc_iota_scale',
        'hm_detection_threshold',
        'threshold_close_pair',
        'scale_to_mm',
        'max_epi_distance',
        'limb_seq',
        'limb_map_idx',
        'sensible_limb_length',
        'min_symmetric_distance'
        'symmetric_joints',
        'ms_radius',
        'ms_sigma',
        'ms_max_iterations'
    ])
    assert len(sensible_limb_length) == len(limb_seq)
    params.ms_radius = ms_radius/scale_to_mm
    if ms_sigma is None:
        ms_sigma = params.ms_radius
    params.ms_sigma = ms_sigma
    params.ms_max_iterations = ms_max_iterations
    params.gc_max_radius = gc_max_radius/scale_to_mm
    params.gc_radius = gc_radius/scale_to_mm
    params.min_nbr_joints = min_nbr_joints
    params.gc_iota_scale = gc_iota_scale
    params.hm_detection_threshold = hm_detection_threshold
    params.threshold_close_pair = threshold_close_pair
    params.scale_to_mm = scale_to_mm
    params.max_epi_distance = max_epi_distance
    params.limb_seq = limb_seq
    params.limb_map_idx = limb_map_idx
    params.sensible_limb_length = sensible_limb_length/scale_to_mm
    params.symmetric_joints = symmetric_joints
    params.min_symmetric_distance = min_symmetric_distance/scale_to_mm
    return params
