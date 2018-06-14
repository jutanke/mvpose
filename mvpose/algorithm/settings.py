from collections import namedtuple
from mvpose.data.default_limbs import DEFAULT_LIMB_SEQ, \
    DEFAULT_MAP_IDX, DEFAULT_SENSIBLE_LIMB_LENGTH


def get_settings(min_nbr_joints=8, gc_iota_scale=6,
                 gc_max_radius=300, gc_radius=50,
                 hm_detection_threshold=0.1,
                 threshold_close_pair=10, scale_to_mm=1,
                 max_epi_distance=10,
                 limb_seq=DEFAULT_LIMB_SEQ,
                 limb_map_idx=DEFAULT_MAP_IDX,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH):
    """
        gets the parameters that are needed for the the program
        !!Careful!!: to set the parameters that are defined in world coordinates
        the unit 'mm' MUST BE used. However, inside the setting object this
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
    :param limb_seq: [ (a, b), (a, c), ... ] list of limbs by joint connection
    :param limb_map_idx: maps the limb id to the part affinity field positions
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
        'sensible_limb_length'
    ])
    assert len(sensible_limb_length) == len(limb_seq)
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
    return params
