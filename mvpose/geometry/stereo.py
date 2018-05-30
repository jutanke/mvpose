import cv2
import numpy as np
import numpy.random as rnd
import mvpose.geometry.geometry as gm


def get_fundamental_matrix(P1, P2):
    """
        finds the fundamental matrix between two views
    :param P1: {3x4} projection matrix
    :param P2: {3x4} projection matrix
    :return:
    """
    points3d = rnd.randint(0, 1500, (12, 3)).astype('float32')
    points1 = np.zeros((12, 2))
    points2 = np.zeros((12, 2))
    for i, (x,y,z) in enumerate(points3d):
        p3d = np.array([x,y,z,1])
        a1, b1, c1 = P1 @ p3d
        a2, b2, c2 = P2 @ p3d
        points1[i, 0] = a1/c1
        points1[i, 1] = b1/c1
        points2[i, 0] = a2 / c2
        points2[i, 1] = b2 / c2
    F, mask = cv2.findFundamentalMat(
        points1, points2, cv2.FM_8POINT
    )
    return F


def triangulate(peaks1, peaks2, P1, P2, max_epi_distance):
    """
       triangulates all points in peaks1 with all points
       in peaks2 BUT drops them if the distance in pixels
       to the epipolar line in either of the two views is
       larger then a threshold
    :param peaks1: [ [(x,y,w), ..], [..] ] * n_joints
    :param peaks2: [ [(x,y,w), ..], [..] ] * n_joints
    :param P1: 3x4 projection matrix
    :param P2: 3x4 projection matrix
    :param max_epi_distance: drop triangulation threshold
    :return:
    """
    n_joints = len(peaks1)
    assert n_joints == len(peaks2)

    F = get_fundamental_matrix(P1, P2)
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
