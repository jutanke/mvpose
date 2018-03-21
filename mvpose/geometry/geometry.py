import cv2
import numpy as np


def get_camera_pos_in_world_coords(rvec, tvec):
    """
    Converts the camera-centric position into
    the camera center in world coordinates
    :param rvec: 3x1
    :param tvec: 3x1
    :return: 3x1
    """
    R = cv2.Rodrigues(rvec)[0]
    return -np.transpose(R) @ tvec