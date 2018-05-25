from skimage.feature import peak_local_max
import mvpose.geometry.geometry as gm
import cv2
import numpy as np


def get_all_peaks(heatmap, threshold):
    """
    extracts peaks from the heatmap
    :param heatmap: h x w x m
    :param threshold:
    :return: [
            [ (x,y,score), (x,y,score),... ]  # Nose
            ...
    ]
    """
    _, _, n_joints = heatmap.shape

    peaks = []
    for i in range(n_joints):
        hm = heatmap[:,:,i]
        local_peaks = peak_local_max(hm, threshold_abs=threshold)
        found_peaks = []
        for x,y in local_peaks:
            found_peaks.append((y,x,hm[x,y]))
        peaks.append(np.array(found_peaks))
    return peaks


class Candidates2D:
    """
        Generates 2D candidates for the heatmaps
        for the original image and for the undistorted
        one
    """

    def __init__(self, heatmaps, Calib, threshold=0.1):
        """
                    n camera views, m joint types
        :param heatmaps: np.array: n x h x w x m
        :param Calib:
        :param threshold:
        """
        n, h, w, m = heatmaps.shape
        assert n == len(Calib)

        self.peaks2d = []
        self.peaks2d_undistorted = []

        self.undistort_maps = []
        self.Calib_undistorted = []

        for cid, cam in enumerate(Calib):
            hm = heatmaps[cid]

            peaks = get_all_peaks(hm, threshold)
            self.peaks2d.append(peaks)

            # -- undistort peaks --
            K, rvec, tvec, distCoef = gm.get_camera_parameters(cam)
            hm_ud, K_new = gm.remove_distortion(hm, cam)
            h, w, _ = hm.shape

            mapx, mapy = \
                cv2.initUndistortRectifyMap(
                    K, distCoef, None, K_new, (w, h), 5)
            self.undistort_maps.append((mapx, mapy))

            peaks_undist = []
            for joint in peaks:
                if len(joint) > 0:
                    print('joint:', joint.shape)
                    peaks_undist.append(
                        gm.undistort_points(joint, mapx, mapy)
                    )
                else:
                    peaks_undist.append([])

            assert len(peaks) == len(peaks_undist)
            self.peaks2d_undistorted.append(peaks_undist)

            self.Calib_undistorted.append({
                'K': K_new,
                'distCoeff': 0,
                'rvec': rvec,
                'tvec': tvec
            })