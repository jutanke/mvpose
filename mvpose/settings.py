from collections import namedtuple
from mvpose.data.default_limbs import DEFAULT_LIMB_SEQ, \
    DEFAULT_MAP_IDX, DEFAULT_SENSIBLE_LIMB_LENGTH


def get_settings(min_nbr_joints=7,
                 gp_max_radius=300,
                 hm_detection_threshold=0.1,
                 pp_conflict_overlap=0.4,
                 scale_to_mm=1,
                 max_epi_distance=10,
                 ms_radius=30,
                 ms_max_iterations=1000,
                 ms_between_distance=100,
                 limb_seq=DEFAULT_LIMB_SEQ,
                 limb_map_idx=DEFAULT_MAP_IDX,
                 sensible_limb_length=DEFAULT_SENSIBLE_LIMB_LENGTH,
                 min_joint_distance=50
                 ):
    """

    :param min_nbr_joints: number of joints that are needed
        to be a 'valid' human pose detection
    :param gp_max_radius: maximal radius for internal distances
        of 3d joint candidates in mm
    :param hm_detection_threshold: confidence map threshold for
        when a peak is a detection and when it is an outlier [0 .. 1]
    :param pp_conflict_overlap: [0 .. 1] value that defines after what
        overlap two persons (according to their aabb) are in conflict
        in a camera view
    :param scale_to_mm: multiplier that scales the world coordinates
        into mm. E.g. if the world coordinates are in m, than
        scale_to_mm = 1000
    :param max_epi_distance:  maximum distance to the respective
        epipolar line that allows two points to be triangulated
        (in pixel)
    :param ms_radius: range for the meanshift density estimation (in [mm])
    :param ms_max_iterations: cut-of threshold for meanshift
    :param ms_between_distance: maximal distance between two points of a
        cluster in [mm]
    :param limb_seq: [ (a, b), (a, c), ... ] list of limbs by joint
        connection
    :param limb_map_idx: maps the limb id to the part affinity field
        positions
    :param sensible_limb_length: [ (min, max), (min, max), ... ] defines
        the sensible length in mm
    :param min_joint_distance: {float} minimal allowed distance between
        different joint types in [mm]
    :return:
    """
    params = namedtuple('Settings', [
        'min_nbr_joints', 'gp_max_radius', 'hm_detection_threshold',
        'pp_conflict_overlap', 'scale_to_mm', 'max_epi_distance',
        'ms_radius', 'ms_max_iterations', 'limb_seq', 'sensible_limb_length',
        'min_joint_distance', 'limb_map_idx', 'ms_between_distance'
    ])
    assert len(sensible_limb_length) == len(limb_seq)
    params.ms_radius = ms_radius/scale_to_mm
    params.min_joint_distance = min_joint_distance/scale_to_mm
    params.gp_max_radius = gp_max_radius/scale_to_mm
    params.scale_to_mm = scale_to_mm
    params.hm_detection_threshold = hm_detection_threshold
    params.pp_conflict_overlap = pp_conflict_overlap
    params.max_epi_distance = max_epi_distance
    params.min_nbr_joints = min_nbr_joints
    params.ms_max_iterations = ms_max_iterations
    params.limb_seq = limb_seq
    params.sensible_limb_length = sensible_limb_length
    params.limb_map_idx = limb_map_idx
    params.ms_between_distance = ms_between_distance
    return params
