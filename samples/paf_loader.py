"""
    Helper function to load part affinity fields
    and confidence maps on systems with no GPU
    where realtime processing is not possible
"""
import sys
sys.path.insert(0,'../../easy_multi_person_pose_estimation')
from poseestimation import model
from time import time
import numpy as np
from os.path import isfile, join


class Loader:
    """
    Use this loader instead of the function below if you need to load
    many frames to avoid memory leaks!
    """

    def __init__(self, with_gpu=False):
        self.pe = model.PoseEstimator()
        self.with_gpu = with_gpu

    def load_confidence_map_and_paf(self, name, Im, frame, dir='/tmp'):
        """
            loads the confidence map and paf
        :param name: to store the data
        :param Im: np.array: n x h x w x 3
        :param frame: {int}
        :param with_gpu:
        :param dir
        :return:
        """
        return load_confidence_map_and_paf(
            name, Im, frame, with_gpu=self.with_gpu, dir=dir, pe=self.pe)


def load_confidence_map_and_paf(name, Im, frame, with_gpu=False, dir='/tmp', pe=None):
    """
        loads the confidence map and paf
    :param name: to store the data
    :param Im: np.array: n x h x w x 3
    :param frame: {int}
    :param with_gpu:
    :param dir
    :return:
    """
    if pe is None:
        pe = model.PoseEstimator()
    if with_gpu:
        heatmaps, pafs = pe.predict_pafs_and_heatmaps(Im)
    else:
        hm_file = join(dir, name + 'heatmaps' + str(frame) + '.npy')
        paf_file = join(dir, name + 'pafs' + str(frame) + '.npy')

        if isfile(hm_file) and isfile(paf_file):
            heatmaps = np.load(hm_file)
            pafs = np.load(paf_file)
        else:
            heatmaps = []
            pafs = []
            for im in Im:
                _start = time()
                hm, paf = pe.predict_pafs_and_heatmaps(im)
                heatmaps.append(np.squeeze(hm))
                pafs.append(np.squeeze(paf))
                _end = time()
                print('elapsed:', _end - _start)
            heatmaps = np.array(heatmaps)
            pafs = np.array(pafs)
            np.save(hm_file, heatmaps)
            np.save(paf_file, pafs)

    return heatmaps[:,:,:,0:-1], pafs