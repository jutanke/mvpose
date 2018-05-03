import cv2
import numpy as np
from numba import vectorize, float64
from math import sqrt


@vectorize([float64(float64,float64,float64,float64,float64, float64)])
def point_to_point_distance(x1,y1,z1, x2, y2, z2):
    """
        calculates the point-to-point distance
    :param x1:
    :param y1:
    :param z1:
    :param x2:
    :param y2:
    :param z2:
    :return:
    """
    a = (x2 - x1)**2
    b = (y2 - y1)**2
    c = (z2 - z1)**2
    return sqrt(a + b + c)


@vectorize([float64(float64,float64,float64,float64,float64)])
def line_to_point_distance(a,b,c,x,y):
    return abs(a*x + b*y + c) / sqrt(a**2 + b**2)


def reproject_points_to_2d(pts3d, rvec, tvec, K, w, h, binary_mask=False):
    """

    :param pts3d:
    :param rvec:
    :param tvec:
    :param K:
    :param w:
    :param h:
    :return:
    """
    distCoef = np.zeros((5, 1))  # to match OpenCV API
    Pts3d = pts3d.astype('float64')
    pts2d, _ = cv2.projectPoints(Pts3d, rvec, tvec, K, distCoef)
    pts2d = np.squeeze(pts2d)

    x = pts2d[:, 0]
    y = pts2d[:, 1]

    mask = (x > 0) * 1
    mask *= (x < w) * 1
    mask *= (y > 0) * 1
    mask *= (y < h) * 1

    if not binary_mask:
        mask = np.nonzero(mask)

    return pts2d, mask


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


def distort_points(points, mapx, mapy):
    """
        applies the distortion to the points
    :param points:
    :param mapx:
    :param mapy:
    :return:
    """
    new_points = points.copy()
    try:
        for idx, (x,y,_) in enumerate(points):
            x = int(x)
            y = int(y)
            new_points[idx, 0] = mapx[y,x]
            new_points[idx, 1] = mapy[y,x]
    except ValueError:
        for idx, (x,y) in enumerate(points):
            x = int(x)
            y = int(y)
            new_points[idx, 0] = mapx[y,x]
            new_points[idx, 1] = mapy[y,x]
    return new_points


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
