from poseestimation import model
import pickle
from os.path import join, isfile


class OpenPose:

    def __init__(self, tmp='/tmp'):
        """
        :param tmp:
        """
        self.pe = model.PoseEstimator()
        self.tmp = tmp

    def predict(self, Im, dataset, frame):
        """
        :param Im:
        :param dataset: name of dataset
        :param frame: frame
        :return:
        """
        fname = join(self.tmp,
                     dataset + '_' + str(frame) + '.posepred')
        if isfile(fname):
            with open(fname, 'rb') as fp:
                predictions = pickle.load(fp)
        else:
            predictions = self.pe.predict(Im)

            # flip x and y coordinates
            for frame, preds in enumerate(predictions):
                for pid, pred in enumerate(preds):
                    for jid in range(18):
                        x = pred[jid, 0]
                        y = pred[jid, 1]
                        predictions[frame][pid, jid, 0] = y
                        predictions[frame][pid, jid, 1] = x

            with open(fname, 'wb') as fp:
                pickle.dump(predictions, fp)

        return predictions
