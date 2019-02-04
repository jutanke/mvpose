import unittest
import tensorflow as tf
import numpy as np
import numpy.linalg as la
import mvpose.topdown.projective_optimization as po


class TestHelperFunctions(unittest.TestCase):

    def test_masked_mean_euclidean_distance3d(self):
        A = np.array([
            [10, 0, 0, 0],
            [0, 0, 0, 1]
        ], np.float32)
        B = np.array([
            [0, 0, 0],
            [0, 0, 0]
        ], np.float32)
        _A = tf.convert_to_tensor(A)
        _B = tf.convert_to_tensor(B)
        _d = po.masked_mean_euclidean_distance(_A, _B)
        sess = tf.Session()
        d = sess.run(_d)
        self.assertAlmostEqual(d, 0)

        A = np.array([
            [10, 0, 0, 1],
            [0, 0, 0, 1]
        ], np.float32)
        B = np.array([
            [0, 0, 0],
            [0, 0, 0]
        ], np.float32)
        _A = tf.convert_to_tensor(A)
        _B = tf.convert_to_tensor(B)
        _d = po.masked_mean_euclidean_distance(_A, _B)
        sess = tf.Session()
        d = sess.run(_d)
        self.assertAlmostEqual(d, 5)

    def test_mean_euclidean_distance3d(self):
        A = np.array([
            [10, 0, 0],
            [0, 0, 0]
        ], np.float32)
        B = np.array([
            [0, 0, 0],
            [0, 0, 0]
        ], np.float32)
        _A = tf.convert_to_tensor(A)
        _B = tf.convert_to_tensor(B)
        _d = po.mean_euclidean_distance(_A, _B)
        sess = tf.Session()
        d = sess.run(_d)
        self.assertAlmostEqual(d, 5)

    def test_mean_euclidean_distance2d(self):
        A = np.array([
            [10, 0],
            [0, 0]
        ], np.float32)
        B = np.array([
            [0, 0],
            [0, 0]
        ], np.float32)
        _A = tf.convert_to_tensor(A)
        _B = tf.convert_to_tensor(B)
        _d = po.mean_euclidean_distance(_A, _B)
        sess = tf.Session()
        d = sess.run(_d)
        self.assertAlmostEqual(d, 5)

    def test_from_homogeneous(self):
        n = 15
        pts3d = np.random.random((n, 4)) - 0.5
        pts3d[:, 3] += 2  # make sure its not 0
        _pts3d = tf.convert_to_tensor(pts3d)
        _pts3d_clear = po.from_homogenous(_pts3d)
        sess = tf.Session()
        pts3d_clear = sess.run(_pts3d_clear)
        H = np.expand_dims(pts3d[:, 3], axis=1)
        pts3d_dehom = pts3d/H
        diff = np.sum(np.abs(pts3d_dehom[:, 0:3] - pts3d_clear))
        self.assertAlmostEqual(0, diff)

    def test_to_homogeneous(self):
        n = 20
        pts3d = np.random.random((n, 3)) - 0.5
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

        _pts3d_back = po.from_homogenous(_pts3d_h)
        pts3d_back = sess.run(_pts3d_back)
        self.assertEqual(n, pts3d_back.shape[0])
        self.assertEqual(3, pts3d_back.shape[1])
        diff = np.sum(np.abs(pts3d_back - pts3d))
        self.assertAlmostEqual(0, diff)


class TestProjection(unittest.TestCase):

    def test_loss_per_cam(self):

        umpm3d = np.random.random((12, 3))
        P = np.random.random((3, 4)).flatten()
        L = np.random.random((12,)).flatten()
        pts2d = np.random.random((12, 2)).flatten()

        y_true = np.concatenate([P, pts2d, L], axis=0).astype(np.float32)
        y_pred = umpm3d.flatten().astype(np.float32)

        _y_true = tf.convert_to_tensor(y_true)
        _y_pred = tf.convert_to_tensor(y_pred)

        _loss = po.loss_per_camera(_y_true, _y_pred)



    def test_simple(self):
        n = 20
        P = np.random.random((3, 4))
        pts3d = np.random.random((n, 3))
        _pts3d = tf.convert_to_tensor(pts3d)
        _P = tf.convert_to_tensor(P)
        _pts2d = po.project_3d_to_2d(_pts3d, _P)
        sess = tf.Session()
        pts2d_tf = sess.run(_pts2d)
        self.assertEqual(2, pts2d_tf.shape[1])
        self.assertEqual(n, pts2d_tf.shape[0])
        pts3d_np_H = np.pad(pts3d, [[0, 0], [0, 1]], 'constant',
                            constant_values=1)
        pts2d_np_H = np.transpose(P @ pts3d_np_H.T)
        pts2d_np_H = pts2d_np_H / np.expand_dims(pts2d_np_H[:, 2], axis=1)
        pts2d_np = pts2d_np_H[:, 0:2]
        diff = np.sum(np.abs(pts2d_np - pts2d_tf))
        self.assertAlmostEqual(0, diff)


if __name__ == '__main__':
    unittest.main()
