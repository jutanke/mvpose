import numpy as np
from scipy.spatial import KDTree
import sklearn.metrics as mt


def cluster(X, between_distance=50, large=99999999):
    """
        clusters the given dataset
    :param X: [ ... (x,y,z,score) ... ]
    :param between_distance: the maximal distance between
            two points of a cluster
    :param large: {float} largest distance value, is used
            to ensure that points are not clustered with
            themselves
    :return:
    """
    Clusters = []

    # we need to pad the distance matrix so that
    # we can ignore the distances of points with
    # themselves
    n = len(X)
    Padding = np.diag([large] * n)

    # create the n x n pairwise distance matrix
    D = mt.pairwise_distances(X[:,0:3], X[:,0:3])
    D += Padding  # see explanation above

    left, right = np.where(D < between_distance)

    ca = [-1] * n  # cluster assignment

    cur_cluster_id = 0

    for a,b in zip(left, right):
        if ca[a] == -1 and ca[b] == -1:
            ca[a] = cur_cluster_id
            ca[b] = cur_cluster_id
            cur_cluster_id += 1
        elif ca[a] == -1:
            ca[a] = ca[b]
        elif ca[b] == -1:
            ca[b] = ca[a]
        else:  # merge clusters!
            for idx in range(n):
                if ca[idx] == ca[b]:
                    ca[idx] = ca[a]

    lookup = dict()  # check at what index a cluster is set
    for i in range(n):
        cid = ca[i]
        if cid in lookup:
            Clusters[lookup[cid]].append(i)
        else:
            Clusters.append([i])
            if cid >= 0:
                lookup[cid] = len(Clusters) - 1

    return Clusters
