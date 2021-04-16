import unittest

import numpy as np

import CoupledLocalFunctions as clf

class TestSupportPoint(unittest.TestCase):
    def test_create(self):
        # some point
        x = np.array([1.0, 4.0, 3.0, -2.0])

        # options for the basis
        basisOptions = dict()
        basisOptions["Type"] = "TotalOrderPolynomials"
        basisOptions["Order"] = 3

        # options for the support point
        ptOptions = dict()
        ptOptions["BasisFunctions"] = "Basis"
        ptOptions["Basis"] = basisOptions
        ptOptions["NumNeighbors"] = 40

        # create the point
        point = clf.SupportPoint(x, ptOptions)

        # make sure the point is in the right place
        self.assertEqual(len(point.x), len(x))
        for i in range(len(x)):
            self.assertAlmostEqual(point.x[i], x[i], places=12)

        # make sure the basis was created
        self.assertEqual(len(point.bases), 1)
        self.assertEqual(point.bases[0].NumBasisFunctions(), 35)
        self.assertEqual(point.numNeighbors[0], 40)
