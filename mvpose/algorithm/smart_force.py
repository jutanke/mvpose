

def estimate(Calib, heatmaps, pafs, settings, debug=False):
    """
        Smarter brute-force approach (still np-hard)
    :param Calib: [ mvpose.geometry.camera, mvpose.geometry.camera, ...] list of n cameras
    :param heatmaps: [n x h x w x j]   // j = #joints
    :param pafs:     [n x h x w x 2*l]  // l = #limbs
    :param settings: parameters for system
    :param debug:
    :return:
    """
    pass