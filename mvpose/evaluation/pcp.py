from collections import namedtuple
import numpy.linalg as la


def evaluate(gt, d, alpha):
    """
        percentage of correctly estimated parts.
        This score only works on single-human estimations
        and the 3d data must be transformed to fit the
        KTH football2 data format (see {transform3d_from_mscoco})
    :param gt: ground truth human
    :param d: detection human
    :param alpha: 0.5
    :return:
    """
    assert len(gt) == 14
    assert len(d) == 14
    result = namedtuple('Result', [
        'upper_arms',
        'lower_arms',
        'lower_legs',
        'upper_legs',
        'pelvis',
        'shoulder',
        'all_parts',
        'head'
    ])

    limbs = [(7, 6), (10, 11)]  # -- lower arms --
    result.lower_arms = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    limbs = [(8, 7), (9, 10)]  # -- upper arms --
    result.upper_arms = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    limbs = [(0, 1), (5, 4)]  # -- lower legs --
    result.lower_legs = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    limbs = [(1, 2), (3, 4)]  # -- upper legs --
    result.upper_legs = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    # limbs = [(2, 3)]  # -- pelvis --
    # result.pelvis = calculate_pcp_for_limbs(alpha, d, gt, limbs)
    #
    # limbs = [(8, 9)]  # -- shoulder --
    # result.shoulder = calculate_pcp_for_limbs(alpha, d, gt, limbs)
    #
    # limbs = [(12, 13)]  # -- head --
    # result.head = calculate_pcp_for_limbs(alpha, d, gt, limbs)
    #
    # limbs = [(7, 6), (10, 11), (8, 7), (9, 10), (0, 1), (5, 4),
    #          (1, 2), (3, 4), (2, 3), (8, 9),
    #          (8, 2), (9, 3)]
    # result.all_parts = calculate_pcp_for_limbs(alpha, d, gt, limbs)
    return result


def calculate_pcp_for_limbs(alpha, d, gt, limbs):
    """
        helper function
    :param alpha:
    :param d:
    :param gt:
    :param limbs:
    :return:
    """
    val = 0
    for a, b in limbs:
        s_hat = gt[a]
        s = d[a]
        e_hat = gt[b]
        e = d[b]
        if s is not None and e is not None:
            s_term = la.norm(s_hat - s)
            e_term = la.norm(e_hat - e)
            term = (s_term + e_term) / 2
            alpha_term = alpha * la.norm(s_hat - e_hat)
            if term <= alpha_term:
                val += 1/len(limbs)
    return val
