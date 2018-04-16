import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
import mvpose.algorithm.meanshift as ms


class TestMeanshift(unittest.TestCase):

    def test_simple(self):

        X = np.array([
            [0, 0, 0, 1],
            [4, 4, 0, 1],
            [4, 0, 0, 1],
            [0, 4, 0, 1]
        ], 'float64')

        x = np.array([0, 0, 0], 'float64')

        center = ms.meanshift(x, X, 5)

        self.assertAlmostEqual(2, center[0], places=2)
        self.assertAlmostEqual(2, center[1], places=2)
        self.assertAlmostEqual(0, center[2], places=2)