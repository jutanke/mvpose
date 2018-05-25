import cv2
import numpy as np
from numba import vectorize, float64, jit, boolean
from math import sqrt


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


def undistort_points(points, mapx, mapy):
    """
    :param points:
    :param mapx:
    :param mapy:
    :return:
    """
    # TODO maybe use numba
    # TODO: This function is wrong!
    new_points = points.copy()
    for idx,(x,y,_) in enumerate(points):
        x = int(x)
        y = int(y)
        dx = x - mapx[y, x]
        dy = y - mapy[y, x]
        new_points[idx, 0] = x + dx
        new_points[idx, 1] = y + dy

    return new_points
