"""
Generate 3D candidates for the problem
"""
import numpy as np
import mvpose.algorithm.heatmaps as mvhm
import mvpose.algorithm.part_affinity_fields as mvpafs
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_MAP_IDX


def generate_candidates(Calib, heatmaps, pafs, limbSeq=DEFAULT_LIMB_SEQ, mapIdx=DEFAULT_MAP_IDX):
    """
        assuming number of joints = {n} and number of limbs = {m}
        generate 3d candidates for a single frame from {p} views
    :param Calib: [{dict}]*{p} list of all cameras:
                [ {"K":3x3, "rvec":3x1, "tvec":3x1, "distCoeff":5x1}, ....]
    :param heatmaps: {np.array[p x h x w x n]}
    :param pafs: {np.array[p x h x w x m]}
    :param limbSeq: {np.array[m x 2]} ids represent the joint (relative to the heatmaps)
    :param mapIdx: {np.array[m x 2]} ids represent the positions in the paf
    :return:
    """
    n_cameras = heatmaps.shape[0]
    n_limbs = limbSeq.shape[0]
    assert len(Calib) == n_cameras
    assert limbSeq.shape == mapIdx.shape
    assert n_cameras == pafs.shape[0]
    assert n_limbs*2 == pafs.shape[3]
    assert limbSeq.shape[1] == 2

    for cid, cam in enumerate(Calib):
        hm = heatmaps[cid]
        peaks = mvhm.get_all_peaks(hm)


        for a,b in limbSeq:
            print("a:" + str(a) + ", b:" + str(b))
