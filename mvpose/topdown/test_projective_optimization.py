import unittest
import tensorflow as tf
import numpy as np
import numpy.linalg as la
import mvpose.topdown.projective_optimization as po


class TestHelperFunctions(unittest.TestCase):

    def test_to_homogeneous(self):
        n = 20
        pts3d = np.random.random((n, 3))
        _pts3d = tf.convert_to_tensor(pts3d)

        _pts3d_h = po.to_homogenous(_pts3d)

        sess = tf.Session()
        pts3d_h = sess.run(_pts3d_h)

        self.assertEqual(n, pts3d_h.shape[0])
        self.assertEqual(4, pts3d_h.shape[1])
        for i in range(n):
            self.assertAlmostEqual(1, pts3d_h[i, 3])

        diff = np.sum(np.abs(pts3d_h[:, 0:3] - pts3d))
        self.assertAlmostEqual(0, diff)


class TestProjection(unittest.TestCase):

    def test_simple(self):
        n = 20
        P = np.random.random((3, 4))
        pts3d = np.random.random((n, 3))




if __name__ == '__main__':
    unittest.main()
