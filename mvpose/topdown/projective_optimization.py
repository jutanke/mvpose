import tensorflow as tf


def to_homogenous(points):
    """ makes the points homogeneous
    :param points: [ n x d]
    :return: [ n x d | 1]
    """
    return tf.pad(points,
                  [(0, 0), (0, 1)],
                  'CONSTANT',
                  constant_values=1)


# def project_3d_to_2d(points3d, P):
#     """
#     :param points3d: {tf.Tensor} [ n x 3]
#     :param P: {tf.Tensor} [3 x 4]
#     :return:
#     """
