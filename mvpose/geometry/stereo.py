import numpy as np
import cv2
from mvpose.geometry import geometry as gm


def get_epipoles_flat(K1, rvec1, tvec1, distCoef1, K2, rvec2, tvec2, distCoef2):
    """
    Calculates the respective epipoles of the two cameras. This is simple:
        calculate the real camera positon of each camera using rvec/tvec and
        then project this centers into the respective images
    :param K1:
    :param rvec1:
    :param tvec1:
    :param distCoef1:
    :param K2:
    :param rvec2:
    :param tvec2:
    :param distCoef2:
    :return:
    """
    pos1 = gm.get_camera_pos_in_world_coords_flat(rvec1, tvec1)
    pos2 = gm.get_camera_pos_in_world_coords_flat(rvec2, tvec2)

    e2 = cv2.projectPoints(np.array([pos2]), rvec1, tvec1, K1, distCoef1)[0]
    e1 = cv2.projectPoints(np.array([pos1]), rvec2, tvec2, K2, distCoef2)[0]

    return np.squeeze(e2), np.squeeze(e1)


def get_epipoles_undistorted(K1, rvec1, tvec1, K2, rvec2, tvec2):
    """
    Get the epipoles assuming no distortion on the cameras
    :param K1:
    :param rvec1:
    :param tvec1:
    :param K2:
    :param rvec2:
    :param tvec2:
    :return:
    """
    return get_epipoles_flat(K1, rvec1, tvec1, 0, K2, rvec2, tvec2, 0)


def get_epipoles(cam1, cam2):
    """
    Calculates the respective epipoles of the two cameras
    :param cam1:
    :param cam2:
    :return:
    """
    K1 = np.array(cam1['K'])
    K2 = np.array(cam2['K'])
    rvec1 = np.array(cam1['rvec'])
    rvec2 = np.array(cam2['rvec'])
    tvec1 = np.array(cam1['tvec'])
    tvec2 = np.array(cam2['tvec'])
    distCoef1 = np.array(cam1['distCoeff'])
    distCoef2 = np.array(cam2['distCoeff'])
    return get_epipoles_flat(K1, rvec1, tvec1, distCoef1, K2, rvec2, tvec2, distCoef2)



