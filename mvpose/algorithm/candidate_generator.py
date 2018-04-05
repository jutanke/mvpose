"""
Generate 3D candidates for the problem
"""
import numpy as np
import mvpose.algorithm.heatmaps as mvhm
import mvpose.algorithm.part_affinity_fields as mvpafs

# assuming 18 joints
DEFAULT_LIMB_SEQ = np.array(
    [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
     [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
     [0,15], [15,17], [2,16], [5,17]])
DEFAULT_LIMB_SEQ.setflags(write=False)  # read-only

DEFAULT_MAP_IDX = np.array(
    [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25],
     [0,  1], [2,  3], [4, 5], [6, 7], [8, 9], [10, 11], [28, 29],
     [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27]])
DEFAULT_MAP_IDX.setflags(write=False)


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
