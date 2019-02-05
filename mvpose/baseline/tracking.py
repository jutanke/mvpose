import numpy as np
from mvpose.baseline.baseline import estimate, distance_between_poses
from scipy.optimize import linear_sum_assignment


def tracking(calib_per_frame, poses_per_frame,
             actual_frames=None,
             epi_threshold=40,
             scale_to_mm=1,
             min_track_length=4,
             max_distance_between_tracks=100):
    """
    :param calib_per_frame: [ [cam1, ... ], ... ] * frames
    :param poses_per_frame: [ [pose1, ...], ... ] * frames
    :param actual_frames: [ frame1, ... ] nd.array {int}
    :param epi_threshold:
    :param scale_to_mm: d * scale_to_mm = d_in_mm
        that means: if our scale is in [m] we need to set
        scale_to_mm = 1000
    :param min_track_length:
    :param max_distance_between_tracks: maximal distance between
        two tracks in [mm]
    :return:
    """
    # check if we only have one set of cameras
    # (cameras do not change over time)
    fixed_cameras = True
    if isinstance(calib_per_frame[0], (list, )):
        fixed_cameras = False
    n_frames = len(poses_per_frame)
    if not fixed_cameras:
        assert n_frames == len(calib_per_frame)
    if actual_frames is not None:
        assert len(actual_frames) == n_frames

    last_seen_delay = 2
    all_tracks = []

    for t in range(n_frames):
        if actual_frames is not None:
            real_t = actual_frames[t]
        else:
            real_t = t

        if fixed_cameras:
            calib = calib_per_frame
        else:
            calib = calib_per_frame[t]

        poses = poses_per_frame[t]
        assert len(poses) == len(calib)

        predictions = estimate(calib, poses,
                               scale_to_mm=scale_to_mm,
                               epi_threshold=epi_threshold)

        possible_tracks = []
        for track in all_tracks:
            if track.last_seen() + last_seen_delay < real_t:
                continue  # track is too old..
            possible_tracks.append(track)

        n = len(possible_tracks)
        if n > 0:
            m = len(predictions)
            D = np.empty((n, m))
            for tid, track in enumerate(possible_tracks):
                for pid, pose in enumerate(predictions):
                    D[tid, pid] = track.distance_to_last(pose)

            rows, cols = linear_sum_assignment(D)
            D = D * scale_to_mm  # ensure that distances in D are in [mm]

            handled_pids = set()
            for tid, pid in zip(rows, cols):
                d = D[tid, pid]
                if d > max_distance_between_tracks:
                    continue

                # merge pose into track
                track = possible_tracks[tid]
                pose = predictions[pid]
                track.add_pose(real_t, pose)
                handled_pids.add(pid)

            # add all remaining poses as tracks
            for pid, pose in enumerate(predictions):
                if pid in handled_pids:
                    continue
                track = Track(real_t, pose,
                              last_seen_delay=last_seen_delay)
                all_tracks.append(track)

        else:  # no tracks yet... add them
            for pose in predictions:
                track = Track(real_t, pose,
                              last_seen_delay=last_seen_delay)
                all_tracks.append(track)

    surviving_tracks = []
    for track in all_tracks:
        if len(track) >= min_track_length:
            surviving_tracks.append(track)

    return surviving_tracks


class Track:

    def __init__(self, t, pose, last_seen_delay):
        """
        :param t: {int} time
        :param pose: 3d * J
        :param last_seen_delay: max delay between times
        """
        self.frames = [int(t)]
        self.poses = [pose]
        self.last_seen_delay = last_seen_delay
        self.lookup = None

    def __len__(self):
        if len(self.frames) == 1:
            return 1
        else:
            first = self.frames[0]
            last = self.frames[-1]
            return last - first + 1

    def last_seen(self):
        return self.frames[-1]

    def add_pose(self, t, pose):
        """ add pose
        :param t:
        :param pose:
        :return:
        """
        last_t = self.last_seen()
        assert last_t < t
        diff = t - last_t
        assert diff <= self.last_seen_delay
        self.frames.append(t)
        self.poses.append(pose)
        self.lookup = None  # reset lookup

    def get_by_frame(self, t):
        """ :returns pose by frame
        :param t:
        :return:
        """
        if self.lookup is None:
            self.lookup = {}
            for t, pose in zip(self.frames, self.poses):
                self.lookup[t] = pose

        if t in self.lookup:
            return self.lookup[t]
        else:
            return None

    def distance_to_last(self, pose):
        """ calculates the distance to the
            last pose
        :param pose:
        :return:
        """
        last_pose = self.poses[-1]
        return distance_between_poses(pose, last_pose)
