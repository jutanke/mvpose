import unittest
import sys
sys.path.insert(0, '../')
from mvpose.candidates.transitivity import TransitivityLookup
import numpy as np

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
