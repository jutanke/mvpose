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
            for i in range(0, n-1):
                cand_i = candidates[i]
                for j in range(1, n):
                    cand_j = candidates[j]

                    conflicts = 0
                    for jid in range(n_joints):
                        if cand_i[jid] is not None and cand_j[jid] is not None:
                            conflicts += 1
                        if conflicts >= 2:
                            break

                    if conflicts == 1: # merge the two candidates if possible
                        cand_new = [None] * n_joints

                        had_conflict_with_distance = False
                        for jid in range(n_joints):
                            a = cand_i[jid]
                            b = cand_j[jid]
                            if a is not None and b is not None:
                                assert len(a) == 4
                                assert len(b) == 4
                                a_ = a[0:3] * a[3]
                                b_ = b[0:3] * b[3]
                                c = (a_ + b_) / (a[3] + b[3])

                                new_point_fits = True
                                # check if this new points fit with all other points
                                for ojid in pair_lookup[jid]:
                                    min_length, max_length = pair_length[jid, ojid]
                                    d1_ok = True
                                    d2_ok = True

                                    pA_other = cand_i[ojid]
                                    if pA_other is not None:
                                        assert len(pA_other) == 4
                                        distance = la.norm(pA_other[0:3] - c)
                                        if min_length > distance or distance > max_length:
                                            d1_ok = False

                                    pB_other = cand_j[ojid]
                                    if pB_other is not None:
                                        distance = la.norm(pB_other[0:3] - c)
                                        if min_length > distance or distance > max_length:
                                            d2_ok = False

                                    # must be connected to at least one
                                    new_point_fits = new_point_fits and d1_ok and d2_ok

                                if new_point_fits:
                                    cand_new[jid] = c
                                else:
                                    had_conflict_with_distance = True
                                    break
                            elif a is not None:
                                assert b is None
                                cand_new[jid] = a
                            elif b is not None:
                                assert a is None
                                cand_new[jid] = b
                            # else both a, b are None

                        if had_conflict_with_distance:
                            print('merge FAILED!!')
                            merged_candidates.append(cand_j)
                            merged_candidates.append(cand_i)
                        else:
                            print('YAY, MERGE SUCCESSFUL')
                            merged_candidates.append(cand_new)

                    else:
                        merged_candidates.append(cand_j)
                        merged_candidates.append(cand_i)

            print('# candidates: ---> ', len(merged_candidates))
            self.merged_candidates = merged_candidates
        else:
            self.merged_candidates = candidates
