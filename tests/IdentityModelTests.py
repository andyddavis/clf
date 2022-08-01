import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestIdentityModel(unittest.TestCase):
    def test_basic(self):
        indim = 4
        outdim = 8

        para = clf.Parameters()
        para.Add("InputDimension", indim)
        para.Add("OutputDimension", outdim)
        para.Add("MaximumOrder", 5)
        para.Add("LocalRadius", 1.0)

        mod = clf.IdentityModel(para)
        self.assertEqual(mod.indim, indim)
        self.assertEqual(mod.outdim, outdim)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        
        self.assertEqual(np.linalg.norm(mod.RightHandSide(x)), 0.0)

        # create a total order multi-index set and the Legendre basis function
        multiSet = clf.MultiIndexSet(para)
        leg = clf.LegendrePolynomials()

        # create the local function
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        func = clf.LocalFunction(multiSet, leg, center, para)
        
        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])

        op = mod.Operator(func, x, coeff)
        fx = func.Evaluate(x, coeff);
        self.assertEqual(len(op), outdim)
        self.assertEqual(len(fx), outdim)
        self.assertAlmostEqual(np.linalg.norm(fx-op), 0.0)

        jac = mod.JacobianWRTCoefficients(func, x, coeff)
        self.assertEqual(np.shape(jac) [0], outdim)
        self.assertEqual(np.shape(jac) [1], func.NumCoefficients())
        jacFD = mod.JacobianWRTCoefficientsFD(func, x, coeff)
        self.assertEqual(np.shape(jacFD) [0], outdim)
        self.assertEqual(np.shape(jacFD) [1], func.NumCoefficients())
        start = 0;
        for i in range(outdim):
            phi = func.featureMatrix.GetFeatureVector(i).Evaluate(x)

            self.assertAlmostEqual(np.linalg.norm(jac[i, 0:start]), 0.0)
            self.assertAlmostEqual(np.linalg.norm(jacFD[i, 0:start]), 0.0)

            self.assertAlmostEqual(np.linalg.norm(jac[i, start:start+len(phi)]-phi), 0.0)
            self.assertAlmostEqual(np.linalg.norm(jacFD[i, start:start+len(phi)]-phi), 0.0)
            start += len(phi)

            self.assertAlmostEqual(np.linalg.norm(jac[i, start:-1]), 0.0)
            self.assertAlmostEqual(np.linalg.norm(jacFD[i, start:-1]), 0.0)
        self.assertAlmostEqual(np.linalg.norm(jac-jacFD), 0.0)

        weights = np.array([random.uniform(-1.0, 1.0) for i in range(outdim)])
        hess = mod.HessianWRTCoefficients(func, x, coeff, weights);
        self.assertEqual(np.shape(hess) [0], func.NumCoefficients())
        self.assertEqual(np.shape(hess) [1], func.NumCoefficients())
        self.assertEqual(np.linalg.norm(hess), 0.0)
        hessFD = mod.HessianWRTCoefficientsFD(func, x, coeff, weights);
        self.assertEqual(np.shape(hessFD) [0], func.NumCoefficients())
        self.assertEqual(np.shape(hessFD) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(hessFD), 0.0)