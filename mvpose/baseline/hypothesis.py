import cv2
import numpy as np
import numpy.linalg as la
import mvpose.geometry.geometry as gm
from mvpose.geometry.stereo import get_fundamental_matrix


def calculate_cost(cam1, person1, cam2, person2):
    """ calculate the epipolar distance between two humans
    :param cam1:
    :param person1:
    :param cam2:
    :param person2:
    :return:
    """
    F = get_fundamental_matrix(cam1.P, cam2.P, 0)

    # drop all points that are -1 -1 (not visible)
    pts1 = []
    pts2 = []
    weights1 = []
    weights2 = []
    for jid in range(18):
        x1, y1, w1 = person1[jid]
        x2, y2, w2 = person2[jid]
        if x1 >= 0 and x2 >= 0:
            pts1.append((x1, y1))
            weights1.append(w1)
            pts2.append((x2, y2))
            weights2.append(w2)
    weights1 = np.clip(weights1, a_min=0, a_max=1)
    weights2 = np.clip(weights2, a_min=0, a_max=1)

    pts1 = np.array(pts1, 'float32')
    pts2 = np.array(pts2, 'float32')

    epilines_1to2 = np.squeeze(
        cv2.computeCorrespondEpilines(pts1, 1, F))

    epilines_2to1 = np.squeeze(
        cv2.computeCorrespondEpilines(pts2, 2, F))

    total = 0

    n_pairs = len(pts1)
    assert n_pairs == len(pts2)

    for p1, l1to2, w1, p2, l2to1, w2 in zip(
            pts1, epilines_1to2, weights1,
            pts2, epilines_2to1, weights2):
        d1 = gm.line_to_point_distance(*l1to2, *p2)
        d2 = gm.line_to_point_distance(*l2to1, *p1)
        total += d1 + d2
    return total / n_pairs  # normalize


class Hypothesis:

    def __init__(self, pts, cam, threshold):
        """
        :param pts: [ (x, y, w), ... ] x18
        :param cam: ProjectiveCamera
        :param threshold: if cost is larger then this
            value then the 'other' must not be merged
        """
        self.points = [pts]
        self.cams = [cam]
        self.threshold = threshold

    def size(self):
        return len(self.points)

    def get_3d_person(self):
        assert self.size() > 1
        # ===================
        n = self.size()
        m = 18
        Points_undist = np.array([[(-1, -1, 1)] * m] * n)
        for cid, (human, cam) in enumerate(zip(self.points, self.cams)):
            lookup = []
            points2d_distorted = []
            for jid, pt in enumerate(human):
                if pt[0] >= 0:
                    points2d_distorted.append(pt)
                    lookup.append(jid)

            points2d_distorted = np.array(points2d_distorted)
            pts_undist = cam.undistort_points(points2d_distorted)
            for jid, pt in zip(lookup, pts_undist):
                Points_undist[cid, jid] = pt
                Points_undist[cid, jid, 2] = 1

        Points_undist = Points_undist.astype('float32')

        # ===================
        points3d = []
        for jid in range(18):
            points2d = []
            weights = []
            cam_ids = []
            for cid, human2d in enumerate(Points_undist):
                u, v, w = human2d[jid]
                if u >= 0:
                    points2d.append((u, v))
                    weights.append(w)
                    cam_ids.append(cid)

            if len(cam_ids) > 1:
                Pts1 = []
                Pts2 = []
                P1s = []
                P2s = []
                n_points = len(cam_ids)
                for i in range(n_points - 1):
                    for j in range(i+1, n_points):
                        Pts1.append(points2d[i])
                        Pts2.append(points2d[j])
                        P1s.append(self.cams[cam_ids[i]].P)
                        P2s.append(self.cams[cam_ids[j]].P)

                all_3d = []
                for pt1, pt2, P1, P2 in zip(Pts1, Pts2, P1s, P2s):
                    _pt1 = np.transpose([pt1])
                    _pt2 = np.transpose([pt2])
                    pt3d = np.squeeze(
                        gm.from_homogeneous(
                            np.transpose(
                                cv2.triangulatePoints(P1, P2, _pt1, _pt2))))
                    all_3d.append(pt3d)

                pt3d = np.mean(all_3d, axis=0)
                points3d.append(pt3d)
            else:
                points3d.append(None)

        return points3d

    # def get_3d_person(self):
    #     """ returns a plotable 3D person
    #     :return:
    #     """
    #     assert self.size() > 1
    #     points3d = []
    #     for jid in range(18):
    #         y = []
    #         cam_ids = []
    #         for cid, human2d in enumerate(self.points):
    #             # TODO make it weighted
    #             u, v, w = human2d[jid]
    #             if u >= 0:
    #                 y.append(u)
    #                 y.append(v)
    #                 y.append(1)  # homogeneous coordinates
    #                 cam_ids.append(cid)
    #         if len(cam_ids) > 1:
    #             assert len(y) % 3 == 0
    #             y = np.array(y)
    #
    #             # solve for X in least-square sense
    #             H = []
    #             for cid in cam_ids:
    #                 H.append(self.cams[cid].P)
    #
    #             H = np.concatenate(H, axis=0)
    #             _x, _y, _z, _h = la.pinv(H) @ y
    #             X_hat = np.array([
    #                 _x/_h,
    #                 _y/_h,
    #                 _z/_h
    #             ])
    #
    #             points3d.append(X_hat)
    #
    #         else:
    #             points3d.append(None)
    #
    #     return points3d

    def calculate_cost(self, o_points, o_cam):
        """
        :param o_points: other points x18
        :param o_cam: other camera
        :return:
        """
        veto = False  # if true we cannot join {other} with this
        total_cost = 0
        for person, cam in zip(self.points, self.cams):
            cost = calculate_cost(cam, person,
                                  o_cam, o_points)
            total_cost += cost
            if cost > self.threshold:
                veto = True

        return total_cost / len(self.points), veto

    def merge(self, o_points, o_cam):
        """ integrate {other} into our hypothesis
        :param o_points:
        :param o_cam:
        :return:
        """
        self.cams.append(o_cam)
        self.points.append(o_points)
