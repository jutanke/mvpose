from mvpose.pose import validate_input


def track(Calib, heatmaps, pafs, settings=None):
    """

    :param Calib: [ [ {mvpose.geometry.camera}, .. ] * n_cameras ] * n_frames
    :param heatmaps: [ [n x h x w x j], ... ] * n_frames
    :param pafs: [ [n x h x w x 2*l], ... ] * n_frames
    :param settings: parameters for system
    :return:
    """
    n_frames = len(heatmaps)
    assert n_frames > 1
    assert len(pafs) == n_frames

    # fix Calib: in some environments the cameras are fixed and do not
    # change: we then only get a single list of {mvpose.geometry.camera}'s
    # but in other instances we might have dynamic cameras where we have
    # a different calibration for each camera and frame. To be able to
    # handle both cases we 'equalize' them here by simply repeating the
    # same cameras if applicable
    if len(Calib) == 1:
        Calib_ = []
        for _ in range(n_frames):
            Calib_.append(Calib[0])
        Calib = Calib_

    settings = validate_input(Calib[0], heatmaps[0], pafs[0], settings)


