"""! @brief Test the local residual (clf::LocalResidual) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestLocalResidual(unittest.TestCase):
    """! @brief Test the local residual (clf::LocalResidual) python interface"""

    def setUp(self):
        """! Set up the input and output dimension"""
        ## The input dimension
        self.indim = 3
        ## The output dimension
        self.outdim = 2

    def tearDown(self):
        """! Test the local residual python interface"""
        radius = 0.1
        numPoints = 100
        point = [random.uniform(-1.0, 1.0) for i in range(self.indim)]
        domain = clf.Hypercube(point-np.array([radius]*self.indim), point+np.array([radius]*self.indim))

        para = clf.Parameters()
        para.Add('InputDimension', self.indim)
        para.Add('OutputDimension', self.outdim)
        para.Add('NumLocalPoints', numPoints)

        maxOrder = 4
        multiSet = clf.MultiIndexSet(self.indim, maxOrder)
        leg = clf.LegendrePolynomials()
        func = clf.LocalFunction(multiSet, leg, domain, para)

        resid = clf.LocalResidual(func, self.system, para);
        self.assertEqual(resid.InputDimension(), func.NumCoefficients())
        self.assertEqual(resid.OutputDimension(), self.outdim*numPoints)
        self.assertEqual(resid.NumPoints(), numPoints)

        coeff = [random.uniform(-1.0, 1.0) for i in range(resid.InputDimension())]

        fx = resid.Evaluate(coeff)
        self.assertEqual(len(fx), resid.OutputDimension())
        start = 0
        for i in range(numPoints):
            self.assertAlmostEqual(np.linalg.norm(fx[start:start+self.outdim]-self.mat@func.Evaluate(resid.GetPoint(i).x, coeff)), 0.0)
            start += self.outdim

        jac = resid.Jacobian(coeff)
        self.assertEqual(np.shape(jac) [0], self.outdim*numPoints)
        self.assertEqual(np.shape(jac) [1], func.NumCoefficients())
        jacFD = resid.JacobianFD(coeff)
        self.assertEqual(np.shape(jacFD) [0], self.outdim*numPoints)
        self.assertEqual(np.shape(jacFD) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(jac-jacFD)/np.linalg.norm(jac), 0.0)

        weights = [random.uniform(-1.0, 1.0) for i in range(resid.OutputDimension())]
        hess = resid.Hessian(coeff, weights)
        self.assertEqual(np.shape(hess) [0], func.NumCoefficients())
        self.assertEqual(np.shape(hess) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(hess), 0.0)
        hessFD = resid.HessianFD(coeff, weights)
        self.assertEqual(np.shape(hessFD) [0], func.NumCoefficients())
        self.assertEqual(np.shape(hessFD) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(hessFD), 0.0)

    def test_identity_model(self):
        """! Test using and identity model"""
        ## The matrix that defines the model
        self.mat = np.identity(self.outdim);
        ## The model
        self.system = clf.IdentityModel(self.indim, self.outdim)

    def test_linear_model(self):
        """! Test using and linear model"""
        ## The matrix that defines the model
        self.mat = np.zeros((self.outdim, self.outdim));
        for i in range(self.outdim):
            for j in range(self.outdim):
                self.mat[i, j] = random.uniform(-1.0, 1.0)

        ## The model
        self.system = clf.LinearModel(self.indim, self.mat)
