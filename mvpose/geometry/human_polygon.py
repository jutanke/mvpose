from shapely.geometry import Polygon
import numpy as np


def polygonize(human2d):
    xy = []

    for pt in human2d:
        if pt is not None:
            xy.append(pt)
    xy = np.array(xy)

    top = np.argmin(xy[:, 1])
    bottom = np.argmax(xy[:, 1])
    left = np.argmin(xy[:, 0])
    right = np.argmax(xy[:, 0])

    poly = Polygon([xy[top], xy[left], xy[bottom], xy[right]])

    # order = [
    #     7, 6, 5, 17, 15, 0, 14, 16, 2, 3, 4, 8, 9, 10,
    #     13, 12, 11, 7
    # ]
    # for o in order:
    #     pt = human2d[o]
    #     if pt is not None:
    #         xy.append(pt)
    # poly = Polygon(xy)
    # area = np.sqrt(poly.area) / 10
    # poly = Polygon(poly.buffer(area).exterior)
    return poly


def intersection_area(poly1, poly2):
    """
    :param poly1:
    :param poly2:
    :return:
    """
