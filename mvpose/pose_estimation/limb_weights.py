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
            nA = lookup[a, 1] - lookup[a, 0]
            nB = lookup[b, 1] - lookup[b, 0]
            total += nA * nB

        assert total == W.shape[0]  # sanity check

        cur_item = 0
        for i in range(len(limbSeq)):
            a = limbSeq[i, 0]
            b = limbSeq[i, 1]
            nA = lookup[a, 1] - lookup[a, 0]
            nB = lookup[b, 1] - lookup[b, 0]
            length = nA * nB

            assert cur_item + length <= total
            data = W[cur_item:cur_item + length].reshape((nA, nB))
            data.setflags(write=False)
            self.cost_by_limb.append(data)
            cur_item += length

        assert len(self.cost_by_limb) == len(limbSeq)

    def __getitem__(self, lid):
        return self.cost_by_limb[lid]