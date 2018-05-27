import cv2
import numpy as np
from numba import vectorize, float64, jit, boolean
from math import sqrt


def reproject_points_to_2d(pts3d, rvec, tvec, K, w, h,
                           distCoef = np.zeros((5, 1)),binary_mask=False):
    """
    :param pts3d:
    :param rvec:
    :param tvec:
    :param K:
    :param w:
    :param h:
    :param distCoef:to match OpenCV API
    :return:
    """
    Pts3d = pts3d.astype('float64')
    pts2d, _ = cv2.projectPoints(Pts3d, rvec, tvec, K, distCoef)
    pts2d = np.squeeze(pts2d)
    if len(pts2d.shape) == 1:
        pts2d = np.expand_dims(pts2d, axis=0)

    x = pts2d[:, 0]
    y = pts2d[:, 1]

    mask = (x > 0) * 1
    mask *= (x < w) * 1
    mask *= (y > 0) * 1
    mask *= (y < h) * 1

    if not binary_mask:
        mask = np.nonzero(mask)

    return pts2d, mask


@vectorize([float64(float64,float64,float64,float64,float64)])
def line_to_point_distance(a,b,c,x,y):
    return abs(a*x + b*y + c) / sqrt(a**2 + b**2)


def get_projection_matrix(K, rvec, tvec):
    """
    generate the projection matrix from its sub-elements
    :param K: camera matirx
    :param rvec: rodrigues vector
    :param tvec: loc vector
    :return:
    """
    R = cv2.Rodrigues(rvec)[0]
    Rt = np.zeros((3, 4))
    Rt[:, 0:3] = R
    Rt[:, 3] = tvec
    return K @ Rt


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
