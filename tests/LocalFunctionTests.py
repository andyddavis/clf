"""! @brief Test the local function (clf::LocalFunction) python interface"""

import unittest

import random

import numpy as np

import PyCoupledLocalFunctions as clf

class TestLocalFunction(unittest.TestCase):
    """! Test the local function (clf::LocalFunction) python interface"""
    
    def test_evaluation(self):
        """! Test the evaluation of a local function"""
        indim = int(5)
        outdim = int(3)
        maxOrder = int(4)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('OutputDimension', outdim)
        para.Add('MaximumOrder', maxOrder)

        multiSet = clf.MultiIndexSet(para)
        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        domain = clf.Hypercube(center-np.array([0.1]*indim), center+np.array([0.1]*indim))

        func = clf.LocalFunction(multiSet, leg, domain, para)
        self.assertEqual(func.InputDimension(), indim)
        self.assertEqual(func.OutputDimension(), outdim)
        self.assertEqual(func.NumCoefficients(), 378)

        x = center + np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])
        fx = func.Evaluate(x, coeff)
        self.assertAlmostEqual(np.linalg.norm(fx-func.featureMatrix.ApplyTranspose(x, coeff)), 0.0)

        for i in range(10):
            self.assertTrue(domain.Inside(func.SampleDomain()))
