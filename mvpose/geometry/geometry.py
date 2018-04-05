import cv2
import numpy as np
from numba import vectorize, float64
from math import sqrt


@vectorize([float64(float64,float64,float64,float64,float64)])
def line_to_point_distance(a,b,c,x,y):
    return abs(a*x + b*y + c) / sqrt(a**2 + b**2)


def get_camera_parameters(cam):
    """
    helper function
    :param cam:
    :return:
    """
    K = np.array(cam['K'])
    rvec = np.array(cam['rvec'])
    tvec = np.array(cam['tvec'])
    distCoef = np.array(cam['distCoeff'])
    return K, rvec, tvec, distCoef


def remove_distortion(I, cam):
    """
    un-distorts the image
    :param cam:
    :param I:
    :return:
    """
    K = np.array(cam['K'])
    distCoef = np.array(cam['distCoeff'])
    if len(I.shape) == 3:
        h,w,_ = I.shape
    else:
        h,w = I.shape
    alpha = 0  # all pixels are valid
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, distCoef, (w,h), alpha)
    mapx, mapy = cv2.initUndistortRectifyMap(K, distCoef, None, K_new, (w, h), 5)
    return cv2.remap(I, mapx, mapy, cv2.INTER_LINEAR), K_new


def apply_undistortion(I, K, K_new, distCoef):
    """
    un-distorts the image given an already undistorted camera matrix
    :param I:
    :param K:
    :param K_new:
    :param distCoef:
    :return:
    """
    w,h,_ = I.shape
    mapx, mapy = cv2.initUndistortRectifyMap(K, distCoef, None, K_new, (w, h), 5)
    return cv2.remap(I, mapx, mapy, cv2.INTER_LINEAR)


def to_homogeneous(x):
    """
    convert to homogeneous coordinate
    :param x:
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if len(x.shape) == 1:
        return np.concatenate([x, [1]])
    else:
        assert len(x.shape) == 2
        n = x.shape[0]
        ones = np.ones((n, 1))
        return np.hstack([x, ones])


def from_homogeneous(x):
    """
    convert from homogeneous coordinate
    :param x:
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if len(x.shape) == 1:
        h = x[-1]
        if h != 0:
            x = x/h
            return x[0:-1]
        else:
            return None
    else:
        assert len(x.shape) == 2
        h = np.expand_dims(x[:, -1], axis=1)
        x = x / h
        return x[:, 0:-1]


def get_camera_pos_in_world_coords_flat(rvec, tvec):
    """
    Converts the camera-centric position into
    the camera center in world coordinates
    :param rvec: 3x1
    :param tvec: 3x1
    :return: 3x1
    """
    R = cv2.Rodrigues(rvec)[0]
    return -np.transpose(R) @ tvec


def get_camera_pos_in_world_coords(cam):
    """
    Converts the camera-centric position into
    the camera center in world coordinates
    :param cam:
    :return:
    """
    rvec = np.array(cam['rvec'])
    tvec = np.array(cam['tvec'])
    return get_camera_pos_in_world_coords_flat(rvec, tvec)


def get_projection_matrix_flat(K, rvec, tvec):
    """
    calculates the projection matrix P
    :param K:
    :param rvec:
    :param tvec:
    :return:
    """
    R = cv2.Rodrigues(rvec)[0]
    Rt = np.zeros((3,4))
    Rt[:,0:3] = R
    Rt[:,3] = tvec
    return K @ Rt


def get_projection_matrix(cam):
    """
    calculates the 3x4 projection matrix P
    :param cam:
    :return:
    """
    rvec = np.array(cam['rvec'])
    K = np.array(cam['K'])
    tvec = np.array(cam['tvec'])
    return get_projection_matrix_flat(K, rvec, tvec)
