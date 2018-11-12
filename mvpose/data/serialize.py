from os.path import isfile, join
from os import remove
import pickle


class DetectionSerializer:

    def __init__(self, names, dir='./'):
        """
        :param names:
        """
        fname = "_".join([str(n) for n in names])
        fname = join(dir, fname + '.pkl')
        self.fname = fname
        if isfile(fname):
            with open(fname, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []

    def store(self, detections, frame):
        """
        :param detections:
        :param frame:
        :return:
        """
        self.data.append((detections, frame))

    def save(self):
        """ dump to file
        :return:
        """
        if isfile(self.fname):
            remove(self.fname)  # delete old saves
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

