import numpy as np
from mvpose.geometry import stereo


class Triangulation:
    """
        triangulates all 2d candidates if the
        epipolar line is close enough to the
        point
    """

    def __init__(self, candidates2d, Calib, max_epi_distance):
        """
        :param candidates2d: {mvpose.algorithm.Candidates2D}
        :param Calib: [cameras]
        :param max_epi_distance {integer} maximal distance
            in pixels from a point pair to their respective
            epipolar line
        """
        assert max_epi_distance > 0
        n_cameras = candidates2d.n_cameras
        n_joints = candidates2d.n_joints
        self.n_cameras = n_cameras
        self.n_joints = n_joints

        Peaks3d = [np.zeros((0, 5))] * n_joints

        for cid1 in range(n_cameras -1):
            peaks1 = candidates2d.peaks2d_undistorted[cid1]
            for cid2 in range(cid1 + 1, n_cameras):
                peaks2 = candidates2d.peaks2d_undistorted[cid2]
                P1 = Calib[cid1].P
                P2 = Calib[cid2].P
                peaks3d = stereo.triangulate(peaks1, peaks2, P1, P2, max_epi_distance=max_epi_distance)
                assert len(peaks3d) == n_joints
                for k in range(n_joints):
                    Peaks3d[k] = np.concatenate([Peaks3d[k], peaks3d[k]], axis=0)

        self.peaks3d_weighted = Peaks3d
