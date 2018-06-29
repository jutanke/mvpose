import cv2
import numpy as np
from numba import vectorize, float64, jit, boolean
from math import sqrt


def aabb_area(aabb):
    """
        calculate area of aabb
    :param aabb: (tx, ty, bx, by)
    :return:
    """
    tx, ty, bx, by = aabb
    assert tx < bx
    assert ty < by
    return (bx - tx) * (by - ty)

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


@jit([float64[:, :](float64[:, :], float64[:, :], float64, float64, boolean)], nopython=True, nogil=True)
def calculate_distance_all4all_opti(A, B, max_distance, min_distance, AB_are_the_same):
    n = len(A)
    m = len(B)
    result_ids = np.zeros((n * m, 3))
    FCTR = 1 if AB_are_the_same else 0  # makes j start counting from 0 every time

    if max_distance == 0 and min_distance == 0:  # no distance min/max range considered!
        cur_pointer = 0
        for i in range(n):
            for j in range(FCTR * (i + 1), m):
                x1 = A[i, 0]
                y1 = A[i, 1]
                z1 = A[i, 2]
                x2 = B[j, 0]
                y2 = B[j, 1]
                z2 = B[j, 2]
                d = point_to_point_distance(x1, y1, z1, x2, y2, z2)
                result_ids[cur_pointer, 0] = i
                result_ids[cur_pointer, 1] = j
                result_ids[cur_pointer, 2] = d
                cur_pointer += 1

        return result_ids[0:cur_pointer]
    else:
        cur_pointer = 0
        for i in range(n):
            for j in range(FCTR*(i+1), m):
                x1 = A[i, 0]
                y1 = A[i, 1]
                z1 = A[i, 2]
                x2 = B[j, 0]
                y2 = B[j, 1]
                z2 = B[j, 2]
                d = point_to_point_distance(x1, y1, z1, x2, y2, z2)
                if min_distance < d < max_distance:
                    result_ids[cur_pointer, 0] = i
                    result_ids[cur_pointer, 1] = j
                    result_ids[cur_pointer, 2] = d
                    cur_pointer += 1

        return result_ids[0:cur_pointer]


def calculate_distance_all4all(A, B, max_distance=0, min_distance=0, AB_are_the_same=False):
    return calculate_distance_all4all_opti(A, B, max_distance, min_distance, AB_are_the_same)


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
