import numpy as np
import numpy.linalg as la


class Peaks:
    """
        stores the peaks in the heatmaps
    """

    def __init__(self, aslist):
        """

        :param aslist:
        """
        self.n_joints = len(aslist)

        total = 0
        for peaks_per_joint in aslist:
            total += len(peaks_per_joint)

        # data is: (x,y,score)
        data = np.zeros((total, 3), 'float64')

        # lookup is organized as follows: (start,end]
        lookup = np.zeros((self.n_joints, 2), 'int32')

        j = 0
        for i, peaks_per_joint in enumerate(aslist):
            lookup[i, 0] = j
            for peak in peaks_per_joint:
                data[j] = peak
                j += 1
            lookup[i, 1] = j

        lookup[-1, 1] -= 1  # the last item

        self.data = data
        self.data.setflags(write=False)
        self.lookup = lookup
        self.lookup.setflags(write=False)

    def get_all(self):
        """
            returns all point independent of the type
        :return:
        """
        return self.data

    def __getitem__(self, jid):
        """
        :param jid: joint id
        :return:
        """
        start, end = self.lookup[jid]
        return self.data[start:end]


class Peaks3D:
    """

    """

    def __init__(self, data):
        """ ctor
            m be the number of joints

        :param data: [{np.array}]: [m x n_i x 4]
        """
        self.n_joints = len(data)
        self.data = data

    def __getitem__(self, jid):
        """
        :param jid: joint id
        :return:
        """
        return self.data[jid]

    def cluster(self, min_distance=20):
        """
            cluster the different joints
        :param min_distance:
        :return:
        """
        for k in range(self.n_joints):
            X = self.data[k]
            n = len(X)
            print("k",k)
            print('\tn:', n)


    def merge(self, other):
        """

        :param other: {Peaks3D}
        :return:
        """
        assert other.n_joints == self.n_joints

        #if simple:
        for j in range(self.n_joints):
            if self.data[j] is None:
                self.data[j] = other.data[j]
            elif other.data[j] is not None:
                self.data[j] = np.concatenate(
                    [self.data[j], other.data[j]], axis=0)
        # else:
        #     for k in range(self.n_joints):
        #         current_data = self.data[k]
        #         other_data = other.data[k]
        #
        #         if current_data is None:
        #             self.data[k] = other_data
        #         elif other_data is not None:
        #             # TODO this is ~O(n^2) but should be vastly improvable
        #             # TODO by using better data structures
        #
        #             POINTS_for_joint = current_data.copy()
        #             for j in range(len(other_data)):
        #                 p_j = other_data[j][0:3]; w_j = other_data[j][3]
        #                 j_got_merged = False
        #                 for i in range(len(POINTS_for_joint)):
        #                     p_i = POINTS_for_joint[i][0:3]; w_i = POINTS_for_joint[i][3]
        #
        #                     dist = la.norm(np.array(p_j - p_i))
        #                     if dist < merge_distance:
        #                         w_sum = w_i + w_j
        #                         a_i = w_i/w_sum
        #
        #                         v = p_j - p_i
        #                         p_new = p_i + v * a_i  # as a_i + a_j = 1
        #                         px,py,pz = p_new
        #                         POINTS_for_joint[i] = np.array([px,py,pz,w_sum])
        #                         j_got_merged = True
        #                         break  # Found a merging point.. SHOULD WE STOP HERE TODO: maybe this is not correct
        #
        #                 if not j_got_merged:  # append to end
        #                     px,py,pz = p_j
        #                     POINTS_for_joint = np.append(POINTS_for_joint, np.array([[px,py,pz,w_j]]), axis=0)
        #
        #
        #                 self.data[k] = POINTS_for_joint
