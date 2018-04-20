import mvpose.geometry.geometry as gm
import mvpose.pose_estimation.heatmaps as mvhm
from mvpose.candidates import peaks as mvpeaks
import mvpose.pose_estimation.part_affinity_fields as mvpafs
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_MAP_IDX
from mvpose.candidates.candidates import Candidates
from mvpose.pose_estimation import extract_human_pose
import cv2


def estimate(Calib, heatmaps, pafs, limbSeq=DEFAULT_LIMB_SEQ, mapIdx=DEFAULT_MAP_IDX):
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
    assert n_limbs * 2 == pafs.shape[3]
    assert limbSeq.shape[1] == 2


    Calib_undist = []
    Peaks_undist = []
    Limb_Weights = []

    for cid, cam in enumerate(Calib):
        hm = heatmaps[cid]
        paf = pafs[cid]
        peaks = mvhm.get_all_peaks(hm)
        limbs = mvpafs.calculate_limb_weights(peaks, paf)
        Limb_Weights.append(limbs)

        K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)

        hm_ud, K_ud = gm.remove_distortion(hm, cam)
        h, w, _ = hm.shape
        mapx, mapy = cv2.initUndistortRectifyMap(K, distCoef, None, K_ud, (w, h), 5)
        peaks_undist = mvpeaks.Peaks.undistort(peaks, mapx, mapy)
        Peaks_undist.append(peaks_undist)

        Calib_undist.append({
            'K': K_ud,
            'distCoeff': 0,
            'rvec': rvec,
            'tvec': tvec
        })

    candidates = Candidates(Peaks_undist, Limb_Weights, Calib_undist, r=200, mode_between_distance=50)
    humans = candidates.humans

    # candidates = Candidates3d()
    # candidates.triangulate(Peaks_undist, Limb_Weights, Calib_undist)
    # modes, W = candidates.calculate_modes(200)
    # humans = extract_human_pose.extract(modes, W, limbSeq)
    return humans