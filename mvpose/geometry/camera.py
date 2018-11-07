"""
    wrapper for different camera types
"""
import numpy as np
import cv2
import mvpose.geometry.geometry as gm


class Camera:

    def __init__(self, P, w, h, cid=-1):
        self.P = P
        self.w = w
        self.h = h
        self.cid = cid  # for debugging

    def undistort(self, im):
        """ undistorts the image
        :param im: {h x w x c}
        :return:
        """
        return im

    def undistort_points(self, points2d):
        """
        :param points2d: [ (x,y,w), ...]
        :return:
        """
        return points2d

    def projectPoints_undist(self, points3d):
        """
            projects 3d points into 2d ones with
            no distortion
        :param points3d: {n x 3}
        :return:
        """
        points2d = np.zeros((len(points3d), 2))
        for i, (x,y,z) in enumerate(points3d):
            p3d = np.array([x, y, z, 1])
            a, b, c = self.P @ p3d
            assert c != 0
            points2d[i, 0] = a/c
            points2d[i, 1] = b/c
            # if c != 0:
            #     points2d[i, 0] = a/c
            #     points2d[i, 1] = b/c
            # else:  # used for affine camera
            #     points2d[i, 0] = a
            #     points2d[i, 1] = b
        return points2d

    def projectPoints(self, points3d, withmask=False, binary_mask=True):
        """
            projects 3d points into 2d with
            distortion being considered
        :param points3d: {n x 3}
        :param withmask: {boolean} if True return mask that tells if a point is in the view or not
        :return:
        """
        pts2d = self.projectPoints_undist(points3d)
        w = self.w
        h = self.h
        if withmask:
            mask = []
            for pt2d in pts2d:
                x = pt2d[0]
                y = pt2d[1]
                if x < 0 or y < 0 or x > w or y > h:
                    mask.append(0)
                else:
                    mask.append(1)

            mask = np.array(mask)
            if not binary_mask:
                mask = mask.nonzero()

            return pts2d, mask
        else:
            return pts2d


class ProjectiveCamera(Camera):
    """
        Projective camera
    """

    def __init__(self, K, rvec, tvec, distCoef, w, h):
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, distCoef, (w, h), 0)
        mapx, mapy = cv2.initUndistortRectifyMap(K, distCoef, None, K_new, (w, h), 5)
        self.K = K
        self.mapx = mapx
        self.mapy = mapy
        self.K_new = K_new
        self.rvec = rvec
        self.tvec = tvec
        self.distCoef = distCoef
        P = gm.get_projection_matrix(K_new, rvec, tvec)
        Camera.__init__(self, P, w, h)

    def get_C(self):
        """
        :return: (x, y, z) of the camera center in world coordinates
        """
        R = cv2.Rodrigues(self.rvec)[0]
        tvec = self.tvec
        return -np.transpose(R) @ tvec

    def undistort(self, im):
        """ undistorts the image
        :param im: {h x w x c}
        :return:
        """
        return cv2.remap(im, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_points(self, points2d):
        """
        :param points2d: [ (x,y,w), ...]
        :return:
        """
        return gm.undistort_points(points2d, self.mapx, self.mapy)

    def projectPoints_undist(self, points3d):
        """
            projects 3d points into 2d ones with
            no distortion
        :param points3d: {n x 3}
        :return:
        """
        pts2d, _ = cv2.projectPoints(points3d,
                                     self.rvec,
                                     self.tvec,
                                     self.K_new, 0)
        pts2d = np.squeeze(pts2d)
        if len(pts2d.shape) == 1:
            pts2d = np.expand_dims(pts2d, axis=0)
        return pts2d

    def projectPoints(self, points3d, withmask=False, binary_mask=True):
        """
            projects 3d points into 2d with
            distortion being considered
        :param points3d: {n x 3}
        :param withmask: {boolean} if True return mask that tells if a point is in the view or not
        :return:
        """
        if withmask:
            return gm.reproject_points_to_2d(
                points3d, self.rvec, self.tvec, self.K, self.w, self.h,
                distCoef=self.distCoef, binary_mask=binary_mask)
        else:
            pts2d, _ = cv2.projectPoints(points3d,
                                         self.rvec,
                                         self.tvec,
                                         self.K, self.distCoef)
            pts2d = np.squeeze(pts2d)
            if len(pts2d.shape) == 1:
                pts2d = np.expand_dims(pts2d, axis=0)
            return pts2d


class AffineCamera(Camera):
    """
        Affine camera
    """

    def __init__(self, P, w, h):
        Camera.__init__(self, P, w, h)