import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
import mvpose.geometry.clustering as clustering


class TestGeometryCluster(unittest.TestCase):

    def test_simple(self):

        X = np.array([
            (0, 0, 0, 1),  # (x,y,z,score)
            (1, 0, 0, 1),
            (1, 1, 0, 1),
            (0, 1, 0, 1),
            (5, 5, 0, 1),
            (5, 4, 0, 1),
            (5, 6, 0, 1)
        ])

        Clusters = clustering.cluster(X, between_distance=1.5)

        self.assertEqual(2, len(Clusters))
        self.assertEqual(4, len(Clusters[0]))
        self.assertEqual(3, len(Clusters[1]))
        for i in range(4):
            self.assertTrue(i in Clusters[0])
        for i in range(4, 7):
            self.assertTrue(i in Clusters[1])

    def test_simple_mix(self):

        X = np.array([
            (0, 0, 0, 1),  # (x,y,z,score)
            (5, 5, 0, 1),
            (5, 4, 0, 1),
            (1, 1, 0, 1),
            (0, 1, 0, 1),
            (1, 0, 0, 1),
            (5, 6, 0, 1)
        ])

        Clusters = clustering.cluster(X, between_distance=1.5)

        self.assertEqual(2, len(Clusters))
        self.assertEqual(4, len(Clusters[0]))
        self.assertEqual(3, len(Clusters[1]))
        for i in [0, 3, 4, 5]:
            self.assertTrue(i in Clusters[0])
        for i in [1, 2, 6]:
            self.assertTrue(i in Clusters[1])

    def test_simple_mix_one_outlier(self):
        X = np.array([
            (0, 0, 0, 1),  # (x,y,z,score)
            (5, 5, 0, 1),
            (5, 4, 0, 1),
            (1, 1, 0, 1),
            (0, 1, 0, 1),
            (1, 0, 0, 1),
            (5, 6, 0, 1),
            (100, 100, 0, 1)
        ])

        Clusters = clustering.cluster(X, between_distance=1.5)

        self.assertEqual(3, len(Clusters))
        self.assertEqual(4, len(Clusters[0]))
        self.assertEqual(3, len(Clusters[1]))
        self.assertEqual(1, len(Clusters[2]))
        for i in [0, 3, 4, 5]:
            self.assertTrue(i in Clusters[0])
        for i in [1, 2, 6]:
            self.assertTrue(i in Clusters[1])
        self.assertTrue(7 in Clusters[2])