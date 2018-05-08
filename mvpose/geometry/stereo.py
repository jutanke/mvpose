import numpy as np
import numpy.random as rnd
import cv2
from mvpose.geometry import geometry as gm
from mvpose.candidates.peaks import Peaks3D
from scipy.optimize import linear_sum_assignment


def get_fundamental_matrix(cam1, cam2):
    """

    :param cam1:
    :param cam2:
    :return:
    """
    K1, rvec1, tvec1, distCoef1 = gm.get_camera_parameters(cam1)
    K2, rvec2, tvec2, distCoef2 = gm.get_camera_parameters(cam2)
    return get_fundamental_matrix_flat(
        K1, rvec1, tvec1, distCoef1,
        K2, rvec2, tvec2, distCoef2
    )


def get_fundamental_matrix_flat(K1, rvec1, tvec1, distCoef1,
                                K2, rvec2, tvec2, distCoef2):
    """
    Calculate the fundamental matrix between two views
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

    # ---
    # TODO: make this work
    # P1 = gm.get_projection_matrix_flat(K1, rvec1, tvec1)
    # P2 = gm.get_projection_matrix_flat(K2, rvec2, tvec2)
    #
    # e2, e1 = get_epipoles_undistorted(K1, rvec1, tvec1,
    #                                   K2, rvec2, tvec2)
    #
    # H = P2 @ la.pinv(P1)
    # ex = mvmath.cross_product_matrix3d(gm.to_homogeneous(e2))
    #
    # return ex @ H

    # --- the 'stupid' way: brute force using opencv
    pos1 = gm.get_camera_pos_in_world_coords_flat(rvec1, tvec1)
    pos2 = gm.get_camera_pos_in_world_coords_flat(rvec2, tvec2)

    top_z = max(pos1[2], pos2[2])
    points3d = rnd.randint(-top_z, top_z, (8, 3)).astype('float32')

    points1 = np.squeeze(
        cv2.projectPoints(points3d, rvec1, tvec1, K1, distCoef1)[0])
    points2 = np.squeeze(
        cv2.projectPoints(points3d, rvec2, tvec2, K2, distCoef2)[0])

    F, mask = cv2.findFundamentalMat(
        points1, points2, cv2.FM_8POINT,
        param1=1
    )

    return F


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


def triangulate_argmax(peaks1, K1, rvec1, tvec1, peaks2, K2, rvec2, tvec2):
    """
        triangulates to sets of points and chooses only the pairs
        for which the points are closest to their epipolar line
    :param peaks1:
    :param K1:
    :param rvec1:
    :param tvec1:
    :param peaks2:
    :param K2:
    :param rvec2:
    :param tvec2:
    :return:
    """
    assert peaks1.n_joints == peaks2.n_joints
    n_joints = peaks1.n_joints

    P1 = gm.get_projection_matrix_flat(K1, rvec1, tvec1)
    P2 = gm.get_projection_matrix_flat(K2, rvec2, tvec2)

    F = get_fundamental_matrix_flat(K1, rvec1, tvec1, 0,
                                    K2, rvec2, tvec2, 0)

    joints_3d = [np.zeros((0, 4))] * n_joints

    idx_pairs_all = [np.zeros((0, 0))] * n_joints

    for k in range(n_joints):
        pts1 = peaks1[k]
        pts2 = peaks2[k]

        if len(pts1) > 0 and len(pts2) > 0:
            epilines_1to2 = np.squeeze(
                cv2.computeCorrespondEpilines(pts1, 1, F))
            if len(epilines_1to2.shape) <= 1:
                epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)

            epilines_2to1 = np.squeeze(
                cv2.computeCorrespondEpilines(pts2, 2, F))
            if len(epilines_2to1.shape) <= 1:
                epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)

            W = np.zeros((len(pts1), len(pts2)))
            Pt1 = []
            Pt2 = []
            idx_pairs = []
            Ep_distances = np.zeros((len(pts1), len(pts2)))

            for idx1, (p1, (a1, b1, c1)) in enumerate(
                    zip(pts1, epilines_1to2)):
                for idx2, (p2, (a2, b2, c2)) in enumerate(
                        zip(pts2, epilines_2to1)):
                    d1 = gm.line_to_point_distance(a1, b1, c1, p2[0], p2[1])
                    d2 = gm.line_to_point_distance(a2, b2, c2, p1[0], p1[1])
                    w1 = p1[2] ** 2
                    w2 = p2[2] ** 2
                    w = (w1 + w2) / 2  # TODO play around with this
                    W[idx1, idx2] = w
                    Ep_distances[idx1, idx2] = (d1 + d2) # TODO play with this
                    Pt1.append(p1[0:2])
                    Pt2.append(p2[0:2])
                    idx_pairs.append((idx1, idx2))

            W = np.array(W)
            Pt1 = np.array(Pt1)
            Pt2 = np.array(Pt2)

            row_idx, col_idx = linear_sum_assignment(Ep_distances)

            Pt1 = np.transpose(Pt1[row_idx])
            Pt2 = np.transpose(Pt2[col_idx])
            W_ = np.array([W[i, j] for i, j in zip(row_idx, col_idx)])
            W_ = np.expand_dims(W_, axis=1)

            pts3d = gm.from_homogeneous(
                np.transpose(cv2.triangulatePoints(P1, P2, Pt1, Pt2)))

            idx_pairs_all[k] = np.array(list(zip(row_idx, col_idx)))
            joints_3d[k] = np.concatenate([pts3d, W_], axis=1)
            assert len(idx_pairs_all[k]) == joints_3d[k].shape[0]

    return joints_3d, idx_pairs_all


def triangulate_with_weights(peaks1, K1, rvec1, tvec1, peaks2, K2, rvec2, tvec2, max_epi_distance=-1):
    """
        Points must be undistorted
    :param peaks1: {Peaks}
    :param K1:
    :param rvec1:
    :param tvec1:
    :param peaks2: {Peaks}
    :param K2:
    :param rvec2:
    :param tvec2:
    :param max_epi_distance: {float} defines the maximal allowed distance in pixels from point to epipolar line
    :return:
    """
    assert peaks1.n_joints == peaks2.n_joints
    n_joints = peaks1.n_joints
    P1 = gm.get_projection_matrix_flat(K1, rvec1, tvec1)
    P2 = gm.get_projection_matrix_flat(K2, rvec2, tvec2)

    F = get_fundamental_matrix_flat(K1, rvec1, tvec1, 0,
                                           K2, rvec2, tvec2, 0)

    joints_3d = [None] * n_joints

    for j in range(n_joints):
        pts1 = peaks1[j]; n = len(pts1)
        pts2 = peaks2[j]; m = len(pts2)

        # (x, y, z, score1, score2, line dist1, line dist2)
        W = []
        Pt1 = []
        Pt2 = []

        if len(pts1) > 0 and len(pts2) > 0:
            epilines_1to2 = np.squeeze(
                cv2.computeCorrespondEpilines(pts1, 1, F))
            if len(epilines_1to2.shape) <= 1:
                epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)

            epilines_2to1 = np.squeeze(
                cv2.computeCorrespondEpilines(pts2, 2, F))
            if len(epilines_2to1.shape) <= 1:
                epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)

            idx = 0
            for p1, (a1, b1, c1) in zip(pts1, epilines_1to2):
                for p2, (a2, b2, c2), in zip(pts2, epilines_2to1):
                    w3 = gm.line_to_point_distance(a1, b1, c1, p2[0], p2[1])
                    w4 = gm.line_to_point_distance(a2, b2, c2, p1[0], p1[1])
                    w1 = p1[2]
                    w2 = p2[2]

                    if max_epi_distance > 0 and (w3 > max_epi_distance or w4 > max_epi_distance):
                        # skip if the distance is too far from the point to epi-line
                        continue

                    W.append((w1, w2, w3, w4))
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
                joints_3d[j] = np.zeros((0, 7))
        else:
            joints_3d[j] = np.zeros((0, 7))

    return joints_3d


def triangulate(peaks1, K1, rvec1, tvec1, peaks2, K2, rvec2, tvec2):
    """
        triangulate
    :param peaks1: {Peaks}
    :param K1:
    :param rvec1:
    :param tvec1:
    :param peaks2: {Peaks}
    :param K2:
    :param rvec2:
    :param tvec2:
    :return:
    """
    assert peaks1.n_joints == peaks2.n_joints
    n_joints = peaks1.n_joints

    P1 = gm.get_projection_matrix_flat(K1, rvec1, tvec1)
    P2 = gm.get_projection_matrix_flat(K2, rvec2, tvec2)

    F = get_fundamental_matrix_flat(K1, rvec1, tvec1, 0,
                                           K2, rvec2, tvec2, 0)

    joints_3d = [None] * n_joints

    idx_pairs_all = [None] * n_joints

    for j in range(n_joints):
        pts1 = peaks1[j]
        pts2 = peaks2[j]

        if len(pts1) > 0 and len(pts2) > 0:
            epilines_1to2 = np.squeeze(
                cv2.computeCorrespondEpilines(pts1, 1, F))
            if len(epilines_1to2.shape) <= 1:
                epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)

            epilines_2to1 = np.squeeze(
                cv2.computeCorrespondEpilines(pts2, 2, F))
            if len(epilines_2to1.shape) <= 1:
                epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)

            # ----
            # TODO this can be easily optimized with numba

            W = []
            Pt1 = []; Pt2 = []
            idx_pairs = []; idx1 = 0
            for p1, (a1, b1, c1) in zip(pts1, epilines_1to2):
                idx2 = 0
                for p2, (a2, b2, c2), in zip(pts2, epilines_2to1):
                    w1 = 1/gm.line_to_point_distance(a1, b1, c1, p2[0], p2[1])
                    w2 = 1/gm.line_to_point_distance(a2, b2, c2, p1[0], p1[1])
                    w3 = p1[2] * p2[2]  # TODO play around with the formulas
                    w = (min(1, w1) + min(1, w2) + w3)/3
                    W.append(w)
                    Pt1.append(p1[0:2])
                    Pt2.append(p2[0:2])
                    idx_pairs.append((idx1, idx2))
                    idx2 += 1
                idx1 += 1

            Pt1 = np.transpose(np.array(Pt1))
            Pt2 = np.transpose(np.array(Pt2))
            W = np.array(W)
            W = np.expand_dims(W, axis=1)

            pts3d = gm.from_homogeneous(
                np.transpose(cv2.triangulatePoints(P1, P2, Pt1, Pt2)))

            joints_3d[j] = np.concatenate([pts3d, W], axis=1)
            idx_pairs_all[j] = np.array(idx_pairs)
            assert len(idx_pairs) == joints_3d[j].shape[0]

    return Peaks3D(joints_3d), idx_pairs_all
