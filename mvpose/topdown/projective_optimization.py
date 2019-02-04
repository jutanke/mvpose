import tensorflow as tf
import numpy as np


def mscoco_to_headless_umpm_3d(detection3d):
    """
    :param detection3d: [ 18 x 3 ] if not detected: None
    :return:
    """
    assert len(detection3d) == 18
    result = np.zeros((12, 3))
    translation = [  # (umpm - 2, coco)
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 11),
        (7, 12),
        (8, 13),
        (9, 8),
        (10, 9),
        (11, 10)
    ]

    # set invalid points to mean position rather then (0, 0, 0)
    # so that the optimization has it easier
    valid_points = []
    invalid_jids = []

    for jid_left, jid_right in translation:
        if detection3d[jid_right] is None:
            invalid_jids.append(jid_left)
            continue
        x, y, z = detection3d[jid_right]
        valid_points.append((x, y, z))
        result[jid_left, 0] = x
        result[jid_left, 1] = y
        result[jid_left, 2] = z

    mean = np.mean(valid_points, axis=0)
    for jid in invalid_jids:
        result[jid] = mean

    return result


def mscoco_to_headless_umpm_2d(detections2d):
    """ convert the mscoco detections into
        headless umpm so that we can run the
        optimization
    :param detections2d: [ n x 18 x 3 ] (x, y, w)
    :return:
    """
    n = len(detections2d)
    assert detections2d.shape[1] == 18
    assert detections2d.shape[2] == 3
    result = np.zeros((n, 12, 3))
    translation = [  # (umpm - 2, coco)
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 11),
        (7, 12),
        (8, 13),
        (9, 8),
        (10, 9),
        (11, 10)
    ]
    for i, detection in enumerate(detections2d):
        for jid_left, jid_right in translation:
            x, y, w = detection[jid_right]
            result[i, jid_left, 0] = x
            result[i, jid_left, 1] = y
            result[i, jid_left, 2] = w if w > 0 else 0

    return result


def loss_per_cameras(y_true, y_pred):
    """
    :param y_true: {batch}
    :param y_pred: {batch}
    :return:
    """
    result = tf.map_fn(
        lambda x: loss_per_camera(x[0], x[1]),
        (y_true, y_pred),
        dtype=tf.float32
    )
    return tf.reduce_mean(result)


def loss_per_camera(y_true, y_pred):
    """ In our case: J = 12
    :param y_true: [ 3 x 4 x J x 3 x J]
        P + 2d location + visibility, length1
    :param y_pred: [ J x 3 ]
    :return:
    """
    J = 12
    dim_2d = 3  # (x, y, vis)
    P = tf.reshape(y_true[0:3*4], (3, 4))
    pts2d_true = tf.reshape(y_true[3*4:3*4 + J * dim_2d], (J, dim_2d))  # x, y, vis
    limb_len_true = y_true[3*4 + J * dim_2d:]

    pts3d = tf.reshape(y_pred, (J, 3))
    pts2d_pred = project_3d_to_2d(pts3d, P)

    distance2d = masked_mean_euclidean_distance(pts2d_true, pts2d_pred)

    umpm_limbs = np.array([
        (2, 3),  # lu arm
        (3, 4),  # ll arm
        (5, 6),  # ru arm
        (6, 7),  # rl arm
        (8, 9),  # lu leg
        (9, 10),  # ll leg
        (11, 12),  # ru leg
        (12, 13),  # rl leg
        (8, 11),  # hip
        (2, 5),  # shoulder
        (2, 8),  # left side
        (5, 11)  # right side
    ]) - 2  # remove first 2 items in UMPM dataset

    limb_len_pred = get_limb_lengths(pts3d, umpm_limbs)
    distance3d = tf.reduce_mean(tf.abs(limb_len_pred - limb_len_true))

    return distance2d + distance3d


def get_limb_lengths(person, limbs):
    """
    :param person: [ J * 3 ]
    :param limbs: {tf.constant} [ (a, b), (b, c), ... ]
    :return:
    """
    distances = tf.map_fn(
        lambda limb: tf.sqrt(
            tf.reduce_sum(
                tf.square(person[limb[0]] - person[limb[1]]))),
        limbs,
        dtype=tf.float32
    )
    return distances


def to_homogenous(points):
    """ makes the points homogeneous
    :param points: [ n x d ]
    :return: [ n x d | 1 ]
    """
    return tf.pad(points,
                  [(0, 0), (0, 1)],
                  'CONSTANT',
                  constant_values=1)


def from_homogenous(points):
    """ converts the values from hom. to non-hom
    :param points: [ n x d ]
    :return:
    """
    H = tf.expand_dims(points[:, -1], axis=1)
    points = points / H
    return points[:, 0:-1]


def project_3d_to_2d(points3d, P):
    """
    :param points3d: {tf.Tensor} [ n x 3]
    :param P: {tf.Tensor} [3 x 4]
    :return:
    """
    points3d_h = to_homogenous(points3d)
    points2d_h = tf.transpose(
        tf.matmul(P, tf.transpose(points3d_h)))
    return from_homogenous(points2d_h)


def masked_mean_euclidean_distance(pts_A_masked, pts_B):
    """
    :param pts_A_masked: {tf.Tensor} [ n x d + 1 ]
    :param pts_B: {tf.Tensor} [ n x d ]
    :return:
    """
    pts_A = pts_A_masked[:, 0:-1]
    visible = tf.expand_dims(pts_A_masked[:, -1], axis=0)
    diff_sq = tf.square(pts_A - pts_B)
    distance = tf.sqrt(tf.reduce_sum(diff_sq, axis=1))
    distance = distance * visible
    return tf.reduce_mean(distance)


def mean_euclidean_distance(pts_A, pts_B):
    """
    :param pts_A: {tf.Tensor} [ n x d ]
    :param pts_B: {tf.Tensor} [ n x d ]
    :return:
    """
    diff_sq = tf.square(pts_A - pts_B)
    distance = tf.sqrt(tf.reduce_sum(diff_sq, axis=1))
    return tf.reduce_mean(distance)
