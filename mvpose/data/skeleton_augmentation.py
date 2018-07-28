"""
    Body structure for skeleton augmentation
    # * neck            (0)
    # * left-shoulder   (1)
    # * left-elbow      (2)
    # * left-hand       (3)
    # * right-shoulder  (4)
    # * right-elbow     (5)
    # * right-hand      (6)
    # * left-hip        (7)
    # * left-knee       (8)
    # * left-foot       (9)
    # * right-hip       (10)
    # * right-knee      (11)
    # * right-foot      (12)
"""
import numpy as np
from keras.models import load_model


def merge_with_mscoco(humans_coco, humans_pred):
    """
        merges the two datasets
    :param humans_coco: output from the end-to-end algorithm
    :param humans_pred: output from the prediction network for
        the limbs
    :return:
    """
    assert len(humans_coco) == len(humans_pred)
    if len(humans_coco) == 0:
        return []

    assert len(humans_coco[0]) == 18
    assert len(humans_pred[0]) == 13

    # left: mscoco position,  right: prediction position
    # if the right side is -1 then we always take coco
    mapping = [(0, -1), (1, 0), (14, -1), (15, -1), (16, -1), (17, -1),
               (2, 4), (3, 5), (4, 6), (5, 1), (6, 2), (7, 3),
               (8, 10), (9, 11), (10, 12), (11, 7), (12, 8), (13, 9)]

    result = []
    for coco, pred in zip(humans_coco, humans_pred):
        current_human = [None] * 18
        for a, b in mapping:
            if b > -1 and coco[a] is None:
                current_human[a] = pred[b]
            else:
                current_human[a] = coco[a]
        result.append(current_human)
    return result


def transform_from_mscoco(humans):
    """
        Transforms the mscoco detection to our structure
    :param humans: [ [(x,y,z), ..., ], ...]
    :return:
    """
    n = len(humans)
    humans_transformed = np.zeros((n, 13, 4))
    for pid, human in enumerate(humans):
        assert len(human) == 18  # mscoco standard

        # we drop the head as our data and the mscoco are
        # not compatible -> mscoco uses eyes, ears and nose
        # while we use only the tip of the head!
        mapping = [(0, 1),
                   (1, 5), (2, 6), (3, 7), (4, 2), (5, 3), (6, 4),  # arms
                   (7, 11), (8, 12), (9, 13), (10, 8), (11, 9), (12, 10)]

        for a, b in mapping:
            pt = human[b]
            if pt is not None:
                humans_transformed[pid, a, 0:3] = pt
                humans_transformed[pid, a, 3] = 1

    return np.array(humans_transformed)


def normalize(indv, scale_to_mm):
    """
        normalize the data to be between -1 and 1
    :param indv:
    :param scale_to_mm:
    :return:
    """
    result = indv.copy()
    n_points, n_dims = indv.shape
    assert n_points == 13
    assert n_dims == 4
    pts = indv[:, 0:3]
    visible = indv[:, 3]
    div = 1000 / scale_to_mm
    mu = np.mean(pts, axis=0)
    result[:, 0:3] = (pts - mu)/div  # because we use mm!
    for i, v in enumerate(visible):
        if v == 0:
            result[i, 0] = 0
            result[i, 1] = 0
            result[i, 2] = 0
    return result, mu


def denormalize(indv, mu, scale_to_mm):
    """
        de-normalizes the data again
    :param indv:
    :param scale_to_mm:
    :return:
    """
    div = 1000 / scale_to_mm
    #mu = np.mean(indv, axis=0)
    return (indv + mu) * div


def plot_indv(ax, indv, visible=np.ones((13,)), color='red'):
    """
        plots the individual onto the axis
    :param ax:
    :param indv:
    :param visible:
    :param color:
    :return:
    """
    for (x, y, z), v in zip(indv, visible):
        if v == 1:
            ax.scatter(x, y, z, color=color)

    limbs = np.array([
        (1, 2), (1, 5), (2, 5),
        (2, 3), (3, 4), (5, 6), (6, 7),
        (2, 8), (5, 11), (8, 11),
        (8, 9), (9, 10), (11, 12), (12, 13)
    ]) - 1
    for a, b in limbs:
        if visible[a] == 1 and visible[b] == 1:
            p_a = indv[a]
            p_b = indv[b]
            ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], [p_a[2], p_b[2]],
                    color=color, alpha=0.8)


class LimbGenerator:
    """
        adds missing limbs for a set of human pose candidates
    """

    def __init__(self, model_path, scale_to_mm,
                 transform_func=transform_from_mscoco,
                 merge_func=merge_with_mscoco):
        """

        :param model_path: path to the model that will predict the data
        :param scale_to_mm: how the used format can be scaled to [mm]
        :param transform_func: takes the input data and transforms it into the skeleton
                format
        :param merge_func: merges the input data and the model data to
                yield the result of this function
        """
        self.model = load_model(model_path)
        self.scale_to_mm = scale_to_mm
        self.transform_func = transform_func
        self.merge_func = merge_func

    def apply(self, humans):
        """
            generates limbs for candidates with missing limbs
        :param humans: [ [ (x,y,z), ... ] * n_limbs ... ] * n_candidates
        :return:
        """
        n = len(humans)
        if n == 0:
            return []
        humans_trans = self.transform_func(humans)
        Mu = []
        for pid, human in enumerate(humans_trans):
            norm, mu = normalize(human, self.scale_to_mm)
            humans_trans[pid] = norm
            Mu.append(mu)

        y_pred = self.model.predict(humans_trans.reshape(n, 13*4)).reshape(n, 13, 3)

        result = np.zeros((n, 13, 3))
        for pid, (human, mu) in enumerate(zip(y_pred, Mu)):
            result[pid] = denormalize(human, mu, self.scale_to_mm)

        return self.merge_func(humans, result)