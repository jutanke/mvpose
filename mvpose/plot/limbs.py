import numpy as np
from mvpose.data.default_limbs import  DEFAULT_LIMB_SEQ, DEFAULT_MAP_IDX
from cselect import color as cs


def plot(ax, im, peaks, limbs, limbSeq=DEFAULT_LIMB_SEQ):
    """
        plots the limbs onto a matplotlib subplot
    :param ax:
    :param im: {w:h:3}
    :param peaks: {Peaks}
    :param limbs: {Limbs}
    :param limbSeq:
    :return:
    """
    ax.imshow(im, alpha=0.3)
    colors = cs.lincolor(len(limbSeq), random_sat=True, random_val=True)
    for idx, (a, b) in enumerate(DEFAULT_LIMB_SEQ):
        candA = peaks[a]
        candB = peaks[b]
        W_limb = limbs[idx]
        nA = len(candA);
        nB = len(candB)
        if nA > 0 and nB > 0:
            for i in range(nA):
                x1, y1 = candA[i][0:2]
                for j in range(nB):
                    x2, y2 = candB[j][0:2]
                    cost = max(W_limb[i, j], 0.1)
                    ax.plot([x1, x2], [y1, y2], c=colors[idx] / 255, linewidth=cost)