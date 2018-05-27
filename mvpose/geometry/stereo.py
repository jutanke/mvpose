import cv2
import numpy as np
import numpy.random as rnd
import mvpose.geometry.geometry as gm


def get_fundamental_matrix(K1, rvec1, tvec1, distCoef1,
                           K2, rvec2, tvec2, distCoef2):
    """
        finds the fundamental matrix between two views
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
    points3d = rnd.randint(0, 1500, (12, 3)).astype('float32')
    points1 = np.squeeze(
        cv2.projectPoints(points3d, rvec1, tvec1, K1, distCoef1)[0])
    points2 = np.squeeze(
        cv2.projectPoints(points3d, rvec2, tvec2, K2, distCoef2)[0])
    F, mask = cv2.findFundamentalMat(
        points1, points2, cv2.FM_8POINT
    )
    return F


def triangulate(peaks1, K1, rvec1, tvec1,
                peaks2, K2, rvec2, tvec2, max_epi_distance):
    """
       triangulates all points in peaks1 with all points
       in peaks2 BUT drops them if the distance in pixels
       to the epipolar line in either of the two views is
       larger then a threshold
    :param peaks1: [ [(x,y,w), ..], [..] ] * n_joints
    :param K1: Camera matrix
    :param rvec1: rodrigues vector
    :param tvec1: loc vector
    :param peaks2: [ [(x,y,w), ..], [..] ] * n_joints
    :param K2: -*-
    :param rvec2: -*-
    :param tvec2: -*-
    :param max_epi_distance: drop triangulation threshold
    :return:
    """
    n_joints = len(peaks1)
    assert n_joints == len(peaks2)

    P1 = gm.get_projection_matrix(K1, rvec1, tvec1)
    P2 = gm.get_projection_matrix(K2, rvec2, tvec2)

    F = get_fundamental_matrix(K1, rvec1, tvec1, 0,
                                    K2, rvec2, tvec2, 0)
    joints_3d = [None] * n_joints

    for j in range(n_joints):
        pts1 = peaks1[j]
        pts2 = peaks2[j]

        # (x, y, z, score1, score2)
        W = []
        Pt1 = []
        Pt2 = []

        if len(pts1) > 0 and len(pts2) > 0:
            epilines_1to2 = np.squeeze(
                cv2.computeCorrespondEpilines(pts1[:, 0:2], 1, F))
            if len(epilines_1to2.shape) <= 1:
                epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)

            epilines_2to1 = np.squeeze(
                cv2.computeCorrespondEpilines(pts2[:, 0:2], 2, F))
            if len(epilines_2to1.shape) <= 1:
                epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)

            for p1, (a1, b1, c1) in zip(pts1, epilines_1to2):
                for p2, (a2, b2, c2), in zip(pts2, epilines_2to1):
                    w3 = gm.line_to_point_distance(a1, b1, c1, p2[0], p2[1])
                    w4 = gm.line_to_point_distance(a2, b2, c2, p1[0], p1[1])
                    w1 = p1[2]
                    w2 = p2[2]

                    if max_epi_distance > 0 and (w3 > max_epi_distance or w4 > max_epi_distance):
                        # skip if the distance is too far from the point to epi-line
                        continue

                    W.append((w1, w2))
                    Pt1.append(p1[0:2])
                    Pt2.append(p2[0:2])

            if len(Pt1) > 0:
                Pt1 = np.transpose(np.array(Pt1))
                Pt2 = np.transpose(np.array(Pt2))
                W = np.array(W)

                pts3d = gm.from_homogeneous(
                    np.transpose(cv2.triangulatePoints(P1, P2, Pt1, Pt2)))

                joints_3d[j] = np.concatenate([pts3d, W], axis=1)
            else:
                joints_3d[j] = np.zeros((0, 5))
        else:
            joints_3d[j] = np.zeros((0, 5))

    return joints_3d
