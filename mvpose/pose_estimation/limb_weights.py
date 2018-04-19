import numpy as np
import numpy.linalg as la


class LimbWeights:
    """
        represents the weight between joints (limbs)
    """

    def __init__(self, W, lookup, limbSeq):
        """
        :param W: calculated by {calculate_weights_for_all}
        :param lookup: extracted from {mvpose.algorithm.peaks}
        :param limbSeq: pre-defined: see {mvpose.algorithm.candidate_generator}
        """
        self.cost_by_limb = []
        total = 0
        for i in range(len(limbSeq)):
            a = limbSeq[i, 0]
            b = limbSeq[i, 1]
            nA = max(0, lookup[a, 1] - lookup[a, 0])
            nB = max(0, lookup[b, 1] - lookup[b, 0])
            total += nA * nB

        assert total == W.shape[0]  # sanity check

        cur_item = 0
        for i in range(len(limbSeq)):
            a = limbSeq[i, 0]
            b = limbSeq[i, 1]
            nA = max(0, lookup[a, 1] - lookup[a, 0])
            nB = max(0, lookup[b, 1] - lookup[b, 0])
            length = nA * nB
               
            assert nA >= 0 and nB >= 0, 'nA=' + str(nA) + ', nB=' + str(nB)
            assert cur_item + length <= total, 'total=' + str(total) +\
                ' but ci+l=' + str(cur_item) + '+' + str(length)
            data = W[cur_item:cur_item + length].reshape((nA, nB))
            data.setflags(write=False)
            self.cost_by_limb.append(data)
            cur_item += length

        assert len(self.cost_by_limb) == len(limbSeq)

    def __getitem__(self, lid):
        return self.cost_by_limb[lid]


class LimbWeights3d:
    """
        represents the limb weights in 3d space
    """

    def __init__(self, peaks3d, all_idx_pairs, limb_pairs, limbSeq,
                 sensible_limb_length, transfer=None):
        """
            Generates the 3d weights for the limbs
        :param peaks3d: ALL 3d points that were generated from
                        triangulation using the 2d peaks
        :param all_idx_pairs: ordered list of index pairs from the
                        triangulation function that were added
                        in order of how the different cameras are
                        combined
        :param limb_pairs: ordered list of limb pairs that were added
                        in order of how the different cameras are
                        combined
        :param limbSeq: limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
        :param sensible_limb_length: {np.array[m x 2]} (low, high) of sensible limb length'
        """
        n_limbs = limbSeq.shape[0]
        assert len(all_idx_pairs) == len(limb_pairs)
        assert limbSeq.shape[1] == 2

        W_all_limbs_last_xy = [(0, 0)] * n_limbs
        W_all_limbs = [None] * n_limbs
        n_per_joints = peaks3d.count_per_joint()  # count the 3d points

        for idx_pairs, (limbs1, limbs2) in zip(all_idx_pairs, limb_pairs):
            # idx_pairs contains the order in which the points where
            # added to peaks3d
            pass

            for lid, (k1, k2) in enumerate(limbSeq):
                W1 = limbs1[lid]
                nA1, nB1 = W1.shape
                pairs1 = idx_pairs[k1]

                W2 = limbs2[lid]
                nA2, nB2 = W2.shape
                pairs2 = idx_pairs[k2]

                mx, my = W_all_limbs_last_xy[lid]

                # https://github.com/jutanke/mvpose/issues/12
                # ~
                if W_all_limbs[lid] is None:
                    n = n_per_joints[k1]
                    m = n_per_joints[k2]
                    W_limb = np.zeros((n, m))
                    W_all_limbs[lid] = W_limb
                else:
                    W_limb = W_all_limbs[lid]
                # ~

                if pairs1 is not None and pairs2 is not None:

                    # if W_all_limbs[lid] is None:
                    #     n = n_per_joints[k1]
                    #     m = n_per_joints[k2]
                    #     W_limb = np.zeros((n, m))
                    #     W_all_limbs[lid] = W_limb
                    # else:
                    #     W_limb = W_all_limbs[lid]

                    assert len(pairs1) == nA1 * nA2
                    assert len(pairs2) == nB1 * nB2

                    for u,(a1,b1) in enumerate(pairs1):
                        for v,(a2,b2) in enumerate(pairs2):

                            # check if points are in range
                            point1 = peaks3d[k1][mx + u][0:3]
                            point2 = peaks3d[k2][my + v][0:3]
                            distance = la.norm(point1 - point2)
                            LENGTH_low, LENGTH_high = sensible_limb_length[lid]
                            if LENGTH_low < distance < LENGTH_high:
                                W_limb[mx+u,my+v] = W1[a1,a2] + W2[b1,b2]

                W_all_limbs_last_xy[lid] = (mx+nA1*nA2, my+nB1*nB2)

        # sanity check:
        for lid, (k1, k2) in enumerate(limbSeq):
            n1,m1 = W_all_limbs_last_xy[lid]
            n2 = n_per_joints[k1]
            m2 = n_per_joints[k2]

            assert n1 == n2, 'expected ' + str(n1) + ' but got ' + str(n2)
            assert m1 == m2, 'expected ' + str(m1) + ' but got ' + str(m2)

        self.W_per_limb = W_all_limbs

    def __getitem__(self, lid):
        """
        :param lid: limb id
        :return:
        """
        return self.W_per_limb[lid]
