import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
import mvpose.geometry.geometry as gm


class TestGeometry(unittest.TestCase):

    def test_point_to_point(self):
        a = np.array([1, 0, 0], 'float64')
        b = np.array([0, 0, 0], 'float64')

        d = gm.point_to_point_distance(*a, *b)

        self.assertAlmostEqual(1, d, places=3)
