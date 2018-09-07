import numpy as np


class ReWeighting:

    def __init__(self, Calib, heatmaps, Points3d):
        """
            Re-weights the points based on re-projection
        :param Calib:
        :param heatmaps:
        :param Points3d: [ n_joints x 4] -> (x,y,z,w)
        """
        n_cameras, h, w, n_joints = heatmaps.shape
        assert len(Points3d) == n_joints
        assert len(Calib) == n_cameras
        for jid in range(n_joints):
            points3d_with_w = Points3d[jid]
            n_points = len(points3d_with_w)
            if n_points > 0:
                W = np.zeros((n_points, 1), 'float32')
                Count = np.zeros((n_points, 1), 'int32')
                points3d = points3d_with_w[:, 0:3]
                for cid, cam in enumerate(Calib):
                    points2d = cam.projectPoints(points3d)
                    points2d = np.round(points2d).astype('int32')
                    for i, (x, y) in enumerate(points2d):
                        if x >= 0 and y >= 0 and x < w and y < h:
                            Count[i] += 1
                            W[i] += heatmaps[cid, y, x, jid]

                W = W/Count
                Points3d[jid][:, 3] = np.squeeze(W)

        self.points3d = Points3d




