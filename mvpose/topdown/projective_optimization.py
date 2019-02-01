import tensorflow as tf
import numpy as np


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
    P = tf.reshape(y_true[0:3*4], (3, 4))
    pts2d_true = tf.reshape(y_true[3*4:3*4 + J * 2], (J, 2))  # x, y, vis
    limb_len_true = y_true[3*4 + J * 2:]

    pts3d = tf.reshape(y_pred, (J, 3))
    pts2d_pred = project_3d_to_2d(pts3d, P)

    distance2d = mean_euclidean_distance(pts2d_true, pts2d_pred)

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


def mean_euclidean_distance(pts_A, pts_B):
    """
    :param pts_A: {tf.Tensor} [ n x d ]
    :param pts_B: {tf.Tensor} [ n x d ]
    :return:
    """
    diff_sq = tf.square(pts_A - pts_B)
    distance = tf.sqrt(tf.reduce_sum(diff_sq, axis=1))
    return tf.reduce_mean(distance)
