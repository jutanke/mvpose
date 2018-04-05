import unittest
import sys
sys.path.insert(0, '../')
import numpy as np
from mvpose.algorithm import peaks as mvpeaks


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
        print(joint2)

# -------------------------------
# RUN IT
# -------------------------------
if __name__ == '__main__':
    unittest.main()