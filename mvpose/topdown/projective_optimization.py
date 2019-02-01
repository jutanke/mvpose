import tensorflow as tf


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

