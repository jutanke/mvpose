import unittest
import sys
sys.path.insert(0, '../')
from mvpose.candidates.transitivity import TransitivityLookup


class TestTransitivity(unittest.TestCase):

    def test_simple(self):
        D = [(0, 0), (0, 1), (1, 0), (1, 1)]
        E_l = [
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 1)
        ]
        E_j = [
            (0, 0, 1),
            (1, 0, 1)
        ]

        tr = TransitivityLookup(D, E_l, E_j)

        intra, intra_choice, inter, inter_choice = tr.query_with_choice(0, 0)

        Intra = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                 for jid, a, b, c in intra]

        Intra_choice = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                        for jid, a, b, c in intra_choice]

        Inter = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                 for jid1, a, jid2, b, jid3, c in inter]

        Inter_choice = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                        for jid1, a, jid2, b, jid3, c in inter_choice]

        self.assertEqual(0, len(Intra))
        self.assertEqual(9, len(Inter))
        self.assertEqual(0, len(Intra_choice))
        self.assertEqual(0, len(Inter_choice))

        D = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        E_l = [
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 1, 2, 0),
            (0, 1, 2, 1)
        ]
        E_j = [
            (0, 0, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, 0, 1)
        ]

        tr = TransitivityLookup(D, E_l, E_j)

        intra, intra_choice, inter, inter_choice = tr.query_with_choice(0, 0)

        Intra = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                 for jid, a, b, c in intra]

        Intra_choice = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                        for jid, a, b, c in intra_choice]

        Inter = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                 for jid1, a, jid2, b, jid3, c in inter]

        Inter_choice = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                        for jid1, a, jid2, b, jid3, c in inter_choice]

        self.assertEqual(len(Intra), len(set(Intra)))
        self.assertEqual(len(Intra_choice), len(set(Intra_choice)))
        self.assertEqual(len(Inter), len(set(Inter)))
        self.assertEqual(len(Inter_choice), len(set(Inter_choice)))
        self.assertEqual(3, len(Intra))
        self.assertEqual(15, len(Inter))
        self.assertEqual(0, len(Intra_choice))
        self.assertEqual(0, len(Inter_choice))

    def test_limb_choice(self):
        D = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        E_l = [
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 1, 2, 0),
            (0, 1, 2, 1)
        ]
        E_j = [
            (0, 0, 1),
            (1, 0, 1)
        ]

        tr = TransitivityLookup(D, E_l, E_j)

        intra, intra_choice, inter, inter_choice = tr.query_with_choice(0, 0)

        Intra = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                 for jid, a, b, c in intra]

        Intra_choice = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                        for jid, a, b, c in intra_choice]

        Inter = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                 for jid1, a, jid2, b, jid3, c in inter]

        Inter_choice = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                        for jid1, a, jid2, b, jid3, c in inter_choice]

        self.assertEqual(len(Intra), len(set(Intra)))
        self.assertEqual(len(Intra_choice), len(set(Intra_choice)))
        self.assertEqual(len(Inter), len(set(Inter)))
        self.assertEqual(len(Inter_choice), len(set(Inter_choice)))
        self.assertEqual(0, len(Intra))
        self.assertEqual(9, len(Inter))
        self.assertEqual(0, len(Intra_choice))
        self.assertEqual(2, len(Inter_choice))

    def test_limb_and_joint_choice(self):
        D = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        E_l = [
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 1, 2, 0),
            (0, 1, 2, 1)
        ]
        E_j = [
            (0, 0, 1),
            (0, 1, 2),
            (1, 0, 1)
        ]

        tr = TransitivityLookup(D, E_l, E_j)

        intra, intra_choice, inter, inter_choice = tr.query_with_choice(0, 0)

        Intra = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                 for jid, a, b, c in intra]

        Intra_choice = [str(tr.lookup[jid, a]) + '-' + str(tr.lookup[jid, b]) + '-' + str(tr.lookup[jid, c]) \
                        for jid, a, b, c in intra_choice]

        Inter = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                 for jid1, a, jid2, b, jid3, c in inter]

        Inter_choice = [str(tr.lookup[jid1, a]) + '-' + str(tr.lookup[jid2, b]) + '-' + str(tr.lookup[jid3, c]) \
                        for jid1, a, jid2, b, jid3, c in inter_choice]

        self.assertEqual(len(Intra), len(set(Intra)))
        self.assertEqual(len(Intra_choice), len(set(Intra_choice)))
        self.assertEqual(len(Inter), len(set(Inter)))
        self.assertEqual(len(Inter_choice), len(set(Inter_choice)))
        self.assertEqual(0, len(Intra))
        self.assertEqual(9, len(Inter))
        self.assertEqual(1, len(Intra_choice))
        self.assertEqual(2, len(Inter_choice))