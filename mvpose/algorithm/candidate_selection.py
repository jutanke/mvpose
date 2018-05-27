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


class CandidateSelector:

    def __init__(self, Humans, Heatmaps, Calib_undistorted,
                 min_nbr_joints,
                 hm_detection_threshold=0.1,
                 threshold_close_pair=10):
        """
        :param Humans: 3d human candidates
        :param Heatmaps:
        :param hm_detection_threshold: threshold after which a
            detection in the confidence map is considered or not
        :param threshold_close_pair: {int} distance in pixels
            after which two points are considered to be close
        """
        n = len(Humans)
        n_cams = len(Calib_undistorted)

        # -1 -> not visible
        #  1 -> visible
        #  2 -> collision
        FLAG_NOT_VISIBLE = -1
        FLAG_VISIBLE = 1
        FLAG_COLLISION = 2
        Visibility_Table = np.zeros((n_cams, n))

        for a in range(n):
            human3d_a = Humans[a]
            n_joints = len(human3d_a)
            for cid, cam in enumerate(Calib_undistorted):
                human2d_a = project_human_to_2d(human3d_a, cam)
                # ==============================================
                # check in how many views two persons co-incide
                # ==============================================
                for b in range(a + 1, n):
                    human3d_b = Humans[b]
                    human2d_b = project_human_to_2d(human3d_b, cam)

                    # check if co-incide
                    distance = calculate2d_proximity(human2d_a, human2d_b)
                    count_close_pairs = 0
                    for d in distance:
                        if d < 0:
                            continue
                        if d < threshold_close_pair:
                            count_close_pairs += 1

                    if count_close_pairs > min_nbr_joints:
                        Visibility_Table[cid, a] = FLAG_COLLISION
                        Visibility_Table[cid, b] = FLAG_COLLISION

                # ==============================================
                # check the heatmap values in all views
                # ==============================================
                hm = Heatmaps[cid]
                h, w, _ = hm.shape
                believe = [-1] * n_joints
                for jid, pt2d in enumerate(human2d_a):
                    if pt2d is not None:
                        x, y = np.around(pt2d).astype('int32')
                        if x > 0 and y > 0 and x < w and y < h:
                            score = hm[y, x, jid]
                            believe[jid] = score

                total = np.sum((np.array(believe) > hm_detection_threshold))
                if total > min_nbr_joints:
                    Visibility_Table[cid, a] = max(FLAG_VISIBLE, Visibility_Table[cid, a])
                else:
                    Visibility_Table[cid, a] = FLAG_NOT_VISIBLE

        Valid_Humans = []
        for human3d, visibility_in_cams in zip(Humans, np.transpose(Visibility_Table)):
            valid_cams = np.sum(visibility_in_cams == 1)
            if valid_cams > 1:  # valid in at least 2 views
                Valid_Humans.append(human3d)

        self.persons = Valid_Humans