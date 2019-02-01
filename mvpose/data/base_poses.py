import numpy as np
import numpy.linalg as la


class BasePoses:
    """ Stores base poses
        n_base_poses x limbs

            0 left upper arm
            1 left lower arm
            2 right upper arm
            3 right lower arm
            4 left upper leg
            5 left lower leg
            6 right upper leg
            7 right lower leg
            8 hip
            9 shoulder
            10 left side
            11 right side
    """

    def __init__(self, base_poses_txt):
        """
        :param base_poses_txt: {n_poses x 12}
        """
        poses = np.loadtxt(base_poses_txt)
        n_poses, n_limbs = poses.shape
        assert n_limbs == 12, str(poses.shape)
        self.normalize_oder = [9, 10, 11, 8, 6, 4, 3, 0]  # in what order should we normalize
        self.poses = poses

    def umpm(self, pose):
        """
        :param pose: {np.array} [14 x 4]  (x, y, z, visible)
                * head (0)                # 0
                * neck (1)
                * left-shoulder (2)
                * left-elbow (3)
                * left-hand (4)
                * right-shoulder (5)      # 5
                * right-elbow (6)
                * right-hand (7)
                * left-hip (8)
                * left-knee (9)
                * left-foot (10)          # 10
                * right-hip (11)
                * right-knee (12)
                * right-foot (13)
        :return:
        """
        n_joints, c = pose.shape
        assert n_joints == 14
        if c == 3:
            pts3d = pose
            visible = [True] * 14
        elif c == 4:
            pts3d = pose[:, 0:3]
            visible = pose[:, 3].astype('int64')
            visible = visible == 1
        else:
            raise NotImplementedError()

        arm_lu = (2, 3)
        arm_ll = (3, 4)
        arm_ru = (5, 6)
        arm_rl = (6, 7)
        hip = (8, 11)
        shoulder = (2, 5)
        leg_lu = (8, 9)
        leg_ll = (9, 10)
        leg_ru = (11, 12)
        leg_rl = (12, 13)
        side_left = (2, 8)
        side_right = (5, 11)

        L = BasePoses.build_length_model(pts3d, visible,
                                         [arm_lu, arm_ll, arm_ru, arm_rl,
                                          leg_lu, leg_ll, leg_ru, leg_rl,
                                          hip, shoulder,
                                          side_left, side_right])

        norm_index = -1
        for n in self.normalize_oder:
            if L[n] > 0:
                norm_index = n
                break

        assert norm_index >= 0
        base_pose = self.find_closest_base_pose(L, norm_index)

        scale = BasePoses.get_scale_factors(L, base_pose, visible, norm_index)

        L_scaled = L * scale
        return L_scaled

    def find_closest_base_pose(self, L, norm_index):
        """
        :param L:
        :param norm_index:
        :return:
        """
        L = L / L[norm_index]
        base_poses = self.poses.copy()
        base_poses = base_poses / np.expand_dims(base_poses[:, norm_index], axis=1)
        diffs = np.sum(np.abs(base_poses - L), axis=1)
        loc = np.argmin(diffs)
        return base_poses[loc]

    @staticmethod
    def get_scale_factors(L, L_base, visible, norm_index):
        """ gets the scale factor for each limb
        :param L: [12x1]
        :param L_base: [12x1]
        :param visible: [12x1] {Boolean}
        :param norm_index: {int} >= 0
        :return:
        """
        assert norm_index >= 0
        scale = np.ones((12, ), np.float32) * -1
        L = L / L[norm_index]  # just in case
        L_base = L_base / L_base[norm_index]
        for lid, (l, l_base, visible) in enumerate(zip(L, L_base, visible)):
            if visible:
                scale[lid] = l_base / l
        return scale

    @staticmethod
    def build_length_model(pose, visible, lookup):
        """

        :param pose:
        :param visible:
        :param lookup:
        :return:
        """
        L = np.ones((12, ), np.float32) * -1
        for lid, (a, b) in enumerate(lookup):
            if visible[a] and visible[b]:
                L[lid] = la.norm(pose[a] - pose[b])

        return L
