from skimage.feature import peak_local_max
import numpy as np
import concurrent.futures


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
    # TODO: this function is slow ~0.3 seconds per call
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


def request_peaks(heatmaps, cid, cam, threshold):
    """

    :param cid:
    :param cam:
    :return:
    """
    hm = heatmaps[cid]

    peaks = get_all_peaks(hm, threshold)

    # -- undistort peaks --
    peaks_undist = []
    for joint in peaks:
        if len(joint) > 0:
            peaks_undist.append(cam.undistort_points(joint))
        else:
            peaks_undist.append([])

    assert len(peaks) == len(peaks_undist)

    return peaks, peaks_undist


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

        self.n_cameras = n
        self.n_joints = m
        self.peaks2d = [None] * n
        self.peaks2d_undistorted = [None] * n

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures_to_peaks = {
                executor.submit(request_peaks, heatmaps, cid, cam, threshold): cid \
                for cid, cam in enumerate(Calib)}
            for future in concurrent.futures.as_completed(futures_to_peaks):
                cid = futures_to_peaks[future]
                print("maybe cid?", cid)
                peaks, peaks_undist = future.result()
                self.peaks2d[cid] = peaks
                self.peaks2d_undistorted[cid] = peaks_undist

        # for cid, cam in enumerate(Calib):
        #     hm = heatmaps[cid]
        #
        #     peaks = get_all_peaks(hm, threshold)
        #     self.peaks2d.append(peaks)
        #
        #     # -- undistort peaks --
        #     peaks_undist = []
        #     for joint in peaks:
        #         if len(joint) > 0:
        #             peaks_undist.append(cam.undistort_points(joint))
        #         else:
        #             peaks_undist.append([])
        #
        #     assert len(peaks) == len(peaks_undist)
        #     self.peaks2d_undistorted.append(peaks_undist)
