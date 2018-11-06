import cv2
import numpy as np
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
        """ returns a plotable 3D person
        :return:
        """
        assert self.size() > 1



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
