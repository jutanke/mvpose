import numpy as np


class SanityChecker:
    """ Drop obviously broken points
    """

    def __init__(self, pts3d, heatmaps, calib, settings):
        """ reprojects all points and drops those that do
            not 'survive'
        :param pts3d:
        :param heatmaps:
        :param calib:
        :param settings:
        """
        assert len(heatmaps) == len(calib)
        detection_threshold = settings.hm_detection_threshold

        survivors = []

        for pt3d_with_w in pts3d:
            pt3d = pt3d_with_w[0:3]
            valid_reprojections = 0
            for cid, cam in enumerate(calib):
                hm = heatmaps[cid]
                h, w = hm.shape
                u, v = np.squeeze(
                    cam.projectPoints(np.array([pt3d], 'float32')))
                u = int(round(u))
                v = int(round(v))
                if 0 <= u and u < w:
                    if 0 <= v and v < h:
                        value = hm[v, u]
                        if value >= detection_threshold:
                            valid_reprojections += 1

            if valid_reprojections > 1:
                survivors.append(pt3d_with_w)

        if len(survivors) > 0:
            self.survivors = np.array(survivors, 'float32')
        else:
            self.survivors = np.zeros((0, 4), 'float32')
