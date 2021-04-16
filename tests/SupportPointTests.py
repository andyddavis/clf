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
        ptOptions["BasisFunctions"] = basisOptions

        # create the point
        point = clf.SupportPoint(x, ptOptions)

        # make sure the point is in the right place
        self.assertEqual(len(point.x), len(x))
        for i in range(len(x)):
            self.assertAlmostEqual(point.x[i], x[i], places=10)

        # make sure the basis was created
        self.assertEqual(point.basis.NumBasisFunctions(), 35)
