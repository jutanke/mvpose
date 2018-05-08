import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from mvpose.geometry import vector_fields as vec
from mvpose.plot.limbs import draw_vector_field
from math import sqrt


class TestVectorFields(unittest.TestCase):

    def testSimple(self):
        U = np.array([
            [1.2, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1.4]
        ], 'float64')

        V = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], 'float64')

        Un, Vn = vec.clamp_to_1(U, V)
        Vec = draw_vector_field(Un, Vn)
        self.assertTrue(np.max(Vec) <= 1)

    def testDifferentdeg(self):
        U = np.array([
            [1, 1, 1, 1],
            [0, 5, 4, 0],
            [0, 0, 0, sqrt(2)]
        ], 'float64')

        V = np.array([
            [0, 0, 0, 0],
            [0, 1, 99, 0],
            [0, 0, 0, sqrt(2)]
        ], 'float64')

        Vec = draw_vector_field(U, V)

        Un, Vn = vec.clamp_to_1(U, V)

        Vec = draw_vector_field(Un, Vn)
        self.assertTrue(np.max(Vec) <= 1)

        n, m = U.shape
        for x in range(n):
            for y in range(m):
                vec1 = np.array([U[x,y], V[x,y]])
                vec2 = np.array([Un[x, y], Vn[x, y]])
                self.assertAlmostEqual(0, np.cross(vec1, vec2))  # still points in the same direction
