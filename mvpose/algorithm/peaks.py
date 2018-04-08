import numpy as np


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


