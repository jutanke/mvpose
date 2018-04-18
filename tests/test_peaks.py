import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from mvpose.candidates import peaks as mvpeaks


class TestPeaks(unittest.TestCase):

    def test_simple(self):
        peaks_list = [
            [(1, 1, 0.5),
            (2, 2, 0.3)],
            [(5, 1, 0.9),
            (1, 5, 0.1)]
        ]

        peaks = mvpeaks.Peaks(peaks_list)
        self.assertEqual(2, peaks.n_joints)

        joint1 = peaks[0]
        self.assertEqual(2, joint1.shape[0])
        self.assertEqual(1, joint1[0,0])

    def test_with_holes(self):
        peaks_list = [
            [(1, 1, 0.5),
            (2, 2, 0.3)],
            [],
            [(5, 1, 0.9),
            (1, 5, 0.1)]
        ]

        peaks = mvpeaks.Peaks(peaks_list)
        self.assertEqual(3, peaks.n_joints)

        joint1 = peaks[0]
        self.assertEqual(2, joint1.shape[0])
        self.assertEqual(1, joint1[0,0])

        joint2 = peaks[1]

    def test_with_holes_at_end(self):
        peaks_list = [
            [(1, 1, 0.5),
            (2, 2, 0.3)],
            [(5, 1, 0.9),
            (1, 5, 0.1)],
            []
        ]

        peaks = mvpeaks.Peaks(peaks_list)
        self.assertEqual(3, peaks.n_joints)

        joint1 = peaks[0]
        self.assertEqual(2, joint1.shape[0])
        self.assertEqual(1, joint1[0,0])

        lookup = peaks.lookup
        nB = lookup[2, 1] - lookup[2, 0]

        self.assertTrue(nB >= 0)


    def test_simple_peaks3d(self):

        data = [
            np.array([
                (1, 0, 0, 0.5),
                (5, 0, 0, 0.6)
            ]),np.array([
                (0, 1, 0, 0.9),
                (0, 5, 0, 0.3)
            ]),np.array([
                (0, 0, 1, 0.6),
                (0, 0, 5, 0.1)
            ])
        ]

        peaks = mvpeaks.Peaks3D(data)
        self.assertEqual(peaks.n_joints, 3)

    def test_merge_peaks3d(self):

        data1 = [
            np.array([
                (1, 0, 0, 0.5),
                (5, 0, 0, 0.6)
            ]), None, np.array([
                (0, 0, 1, 0.6),
                (0, 0, 5, 0.1)
            ])
        ]

        data2 = [
            np.array([
                (10, 0, 0, 0.5),
                (50, 0, 0, 0.6)
            ]), np.array([
                (0, 10, 0, 0.9),
                (0, 50, 0, 0.3)
            ]), np.array([
                (0, 0, 10, 0.6),
                (0, 0, 50, 0.1)
            ])
        ]

        peaks1 = mvpeaks.Peaks3D(data1)
        peaks2 = mvpeaks.Peaks3D(data2)
        self.assertEqual(peaks1.n_joints, 3)
        self.assertEqual(peaks2.n_joints, 3)

        peaks1.merge(peaks2)
        self.assertEqual(len(peaks1[0]), 4)
        self.assertEqual(len(peaks1[1]), 2)
        self.assertEqual(len(peaks1[2]), 4)

    def test_reverse_merge_peaks3d(self):

        data1 = [
            np.array([
                (1, 0, 0, 0.5),
                (5, 0, 0, 0.6)
            ]), None, np.array([
                (0, 0, 1, 0.6),
                (0, 0, 5, 0.1)
            ])
        ]

        data2 = [
            np.array([
                (10, 0, 0, 0.5),
                (50, 0, 0, 0.6)
            ]), np.array([
                (0, 10, 0, 0.9),
                (0, 50, 0, 0.3)
            ]), np.array([
                (0, 0, 10, 0.6),
                (0, 0, 50, 0.1)
            ])
        ]

        peaks1 = mvpeaks.Peaks3D(data1)
        peaks2 = mvpeaks.Peaks3D(data2)
        self.assertEqual(peaks1.n_joints, 3)
        self.assertEqual(peaks2.n_joints, 3)

        peaks2.merge(peaks1)
        self.assertEqual(len(peaks2[0]), 4)
        self.assertEqual(len(peaks2[1]), 2)
        self.assertEqual(len(peaks2[2]), 4)


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()