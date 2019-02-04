from mvpose.baseline.baseline import estimate


def tracking(calib_per_frame, poses_per_frame,
             epi_threshold=40):
    """
    :param calib_per_frame: [ [cam1, ... ], ... ] * frames
    :param poses_per_frame: [ [pose1, ...], ... ] * frames
    :param epi_threshold:
    :return:
    """
    # check if we only have one set of cameras
    # (cameras do not change over time)
    fixed_cameras = True
    if isinstance(poses_per_frame[0], (list, )):
        fixed_cameras = False
    n_frames = len(poses_per_frame)
    if not fixed_cameras:
        assert n_frames == len(calib_per_frame)


    last_seen_delay = 2
    all_tracks = []

    for t in range(n_frames):
        if fixed_cameras:
            calib = calib_per_frame
        else:
            calib = calib_per_frame[t]

        poses = poses_per_frame[t]

        predictions = estimate(calib, poses,
                               epi_threshold=epi_threshold)

        possible_tracks = []
        for track in all_tracks:
            if track.last_seen() + last_seen_delay < t:
                continue  # track is too old..
            possible_tracks.append(track)

        if len(possible_tracks) > 0:
            pass
        else:
            for pose in predictions:
                track = Track(t, pose)
                all_tracks.append(track)



class Track:

    def __init__(self, t, pose):
        """
        :param t: {int} time
        :param pose: 3d * J
        """
        self.frames = [t]
        self.poses = [pose]

    def last_seen(self):
        return self.frames[-1]

    def distance_to_last(self, pose):
        """ calculates the distance to the
            last pose
        :param pose:
        :return:
        """
