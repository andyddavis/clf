import unittest

import random

import numpy as np

import PyCoupledLocalFunctions as clf

class TestLocalFunction(unittest.TestCase):
    def test_basic(self):
        indim = int(5)
        outdim = int(3)
        maxOrder = int(4)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('OutputDimension', outdim)
        para.Add('MaximumOrder', maxOrder)
        para.Add('LocalRadius', 1.0)

        multiSet = clf.MultiIndexSet(para)
        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])

        func = clf.LocalFunction(multiSet, leg, center, para)
        self.assertEqual(func.InputDimension(), indim)
        self.assertEqual(func.OutputDimension(), outdim)
        self.assertEqual(func.NumCoefficients(), 378)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])
        fx = func.Evaluate(x, coeff)
        self.assertAlmostEqual(np.linalg.norm(fx-func.featureMatrix.ApplyTranspose(x, coeff)), 0.0)


