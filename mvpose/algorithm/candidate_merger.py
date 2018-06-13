import numpy.linalg as la


class CandidateMerger:

    def __init__(self, candidates, settings):
        """
            merge two candidates if maximal one of their items overlap
        :param candidates: [ [ (x,y,z,w),... None, ... ], [...] ... ]
        :param settings
        """
        n = len(candidates)
        if n > 1:
            n_joints = len(candidates[0])
            limb_seq = settings.limb_seq
            sensible_limb_length = settings.sensible_limb_length
            pair_lookup = {}
            pair_length = {}
            for lid, (a, b) in enumerate(limb_seq):
                if not a in pair_lookup:
                    pair_lookup[a] = set()
                pair_lookup[a].add(b)
                if not b in pair_lookup:
                    pair_lookup[b] = set()
                pair_lookup[b].add(a)
                pair_length[a, b] = sensible_limb_length[lid]
                pair_length[b, a] = sensible_limb_length[lid]

            merged_candidates = []
            merge_pairs = []  # (i, j, conflict_jid)
            for i in range(0, n-1):
                cand_i = candidates[i]
                for j in range(1, n):
                    cand_j = candidates[j]

                    conflicts = 0
                    conflict_jid = -1
                    for jid in range(n_joints):
                        if cand_i[jid] is not None and cand_j[jid] is not None:
                            conflicts += 1
                            conflict_jid = jid
                        if conflicts >= 2:
                            break

                    if conflicts == 1:
                        merge_pairs.append((i, j, conflict_jid))

                    #
                    # if conflicts == 1: # merge the two candidates if possible
                    #     assert conflict_jid >= 0
                    #     cand_new = [k for k in cand_i]
                    #     k1 = cand_i[conflict_jid]
                    #     k2 = cand_j[conflict_jid]
                    #     k_mix = (k1[0:3] * k1[3] + k2[0:3] * k2[3]) / (k1[3] + k2[3])
                    #     for other_jid in pair_lookup[conflict_jid]:
                    #         min_length, max_length = pair_length[conflict_jid, other_jid]
                    #

        else:
            self.merged_candidates = candidates
