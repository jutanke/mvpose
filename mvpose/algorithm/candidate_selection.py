import numpy as np
import numpy.linalg as la
import cv2
import mvpose.geometry.geometry as gm


def project_human_to_2d(human3d, cam):
    human2d = [None] * len(human3d)
    for jid, pt3d in enumerate(human3d):
        if pt3d is not None:
            Pt = np.array([pt3d])
            K, rvec, tvec, _ = gm.get_camera_parameters(cam)
            points2d = np.squeeze(cv2.projectPoints(Pt, rvec, tvec, K, 0)[0])
            human2d[jid] = points2d
    return human2d


def calculate2d_proximity(person1, person2):
    n_joints = len(person1)
    assert n_joints == len(person2)
    jointwise_proximity = [-1] * n_joints

    for jid, (pt1, pt2) in enumerate(zip(person1, person2)):
        if pt1 is not None and pt2 is not None:
            distance = la.norm(pt1 - pt2)
            jointwise_proximity[jid] = distance
    return jointwise_proximity