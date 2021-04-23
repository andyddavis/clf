import unittest

import numpy as np

from jax import grad

import CoupledLocalFunctions as clf

class TestSupportPoint(unittest.TestCase):
    def test_create(self):
        # some point
        x = np.array([1.0, 4.0, 3.0, -2.0])

        # options for the basis
        basisOptions = dict()
        basisOptions["Type"] = "TotalOrderPolynomials"
        basisOptions["Order"] = 3

        # options for the model
        modelOptions = dict()
        modelOptions['InputDimension'] = len(x)
        model = clf.Model(modelOptions)

        # options for the support point
        ptOptions = dict()
        ptOptions["BasisFunctions"] = "Basis"
        ptOptions["Basis"] = basisOptions
        ptOptions["NumNeighbors"] = 40

        # create the point
        point = clf.SupportPoint(x, model, ptOptions)

        # make sure the point is in the right place
        self.assertEqual(len(point.x), len(x))
        for i in range(len(x)):
            self.assertAlmostEqual(point.x[i], x[i], places=12)

        # make sure the basis was created
        bases = point.GetBasisFunctions()
        self.assertEqual(len(bases), 1)
        self.assertEqual(bases[0].NumBasisFunctions(), 35)
        self.assertEqual(point.NumNeighbors(), 40)
        self.assertEqual(point.model.inputDimension, len(x))
        self.assertEqual(point.model.outputDimension, 1)
