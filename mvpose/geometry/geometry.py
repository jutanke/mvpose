import cv2
import numpy as np


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
    h,w,_ = I.shape
    alpha = 0  # all pixels are valid
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, distCoef, (w,h), alpha)
    mapx, mapy = cv2.initUndistortRectifyMap(K, distCoef, None, K_new, (w, h), 5)
    return cv2.remap(I, mapx, mapy, cv2.INTER_LINEAR), K_new


def to_homogeneous(x):
    """
    convert to homogeneous coordinate
    :param x:
    :return:
    """
    return np.concatenate([x, [1]])


def from_homogeneous(x):
    """
    convert from homogeneous coordinate
    :param x:
    :return:
    """
    h = x[-1]
    if h != 0:
        x = x/h
        return x[0:-1]
    else:
        return None


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
