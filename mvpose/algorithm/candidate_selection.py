import numpy as np
import numpy.linalg as la


def project_human_to_2d(human3d, cam):
    """

    :param human3d: [ (x,y,z) ... ] OR [ (x,y,z,w), ... ]
    :param cam:
    :return:
    """
    human2d = [None] * len(human3d)
    for jid, pt3d in enumerate(human3d):
        if pt3d is not None:
            if len(pt3d) == 3:
                Pt = np.array([pt3d])
            elif len(pt3d) == 4:
                Pt = np.array([pt3d[0:3]])
            else:
                raise ValueError('pt3d invalid:', pt3d.shape)
            points2d = cam.projectPoints_undist(Pt)  # TODO shouldn't this be WITH distortion?
            human2d[jid] = np.squeeze(points2d)
            if len(pt3d) == 4:
                human2d[jid] = np.append(human2d[jid], pt3d[3])

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


def center_distance(human_a, human_b):
    """
        calculates the center distance
    :param human_a: 2d human
    :param human_b:
    :return:
    """
    assert len(human_a) == len(human_b)
    avg_a = []
    avg_b = []
    for i in range(len(human_a)):
        if human_a[i] is not None:
            avg_a.append(human_a[i])
        if human_b[i] is not None:
            avg_b.append(human_b[i])
    avg_a = np.mean(avg_a, axis=0)
    avg_b = np.mean(avg_b, axis=0)
    return la.norm(avg_a - avg_b)


class CandidateSelector:

    def __init__(self, Humans, Heatmaps, Calib,
                 min_nbr_joints,
                 hm_detection_threshold,
                 threshold_close_pair):
        """
        :param Humans: 3d human candidates
        :param Heatmaps:
        :param hm_detection_threshold: threshold after which a
            detection in the confidence map is considered or not
        :param threshold_close_pair: {int} distance in pixels
            after which two points are considered to be close
        """
        n = len(Humans)
        n_cams = len(Calib)

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
            for cid, cam in enumerate(Calib):
                human2d_a = project_human_to_2d(human3d_a, cam)
                # ==============================================
                # check in how many views two persons co-incide
                # ==============================================
                for b in range(a + 1, n):
                    human3d_b = Humans[b]
                    human2d_b = project_human_to_2d(human3d_b, cam)

                    # check if coincidence
                    distance = calculate2d_proximity(human2d_a, human2d_b)
                    count_close_pairs = 0
                    for d in distance:
                        if d < 0:
                            continue
                        if d < threshold_close_pair:
                            count_close_pairs += 1

                    if count_close_pairs >= min_nbr_joints:
                        Visibility_Table[cid, a] = FLAG_COLLISION
                        Visibility_Table[cid, b] = FLAG_COLLISION
                    else:
                        # check if the center points are too close to each other
                        distance = center_distance(human2d_a, human2d_b)
                        if distance < threshold_close_pair:  # in [pixel]
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
                        if len(pt2d.shape) > 1:
                            pt2d = np.squeeze(pt2d)
                        x, y = np.around(pt2d).astype('int32')
                        if x > 0 and x < w and y > 0 and y < h:
                            score = hm[y, x, jid]
                            believe[jid] = score

                total = np.sum((np.array(believe) > hm_detection_threshold))
                if total >= min_nbr_joints:
                    Visibility_Table[cid, a] = max(FLAG_VISIBLE, Visibility_Table[cid, a])
                else:
                    Visibility_Table[cid, a] = FLAG_NOT_VISIBLE

        Valid_Humans = []
        for human3d, visibility_in_cams in zip(Humans, np.transpose(Visibility_Table)):
            valid_cams = np.sum(visibility_in_cams == 1)
            if valid_cams > 1:  # valid in at least 2 views
                Valid_Humans.append(human3d)

        self.persons = Valid_Humans