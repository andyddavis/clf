import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestLinearModel(unittest.TestCase):
    def setUp(self):
        self.indim = 4
        self.outdim = 3

    def tearDown(self):
        self.assertEqual(self.linsys.indim, self.indim)
        self.assertEqual(self.linsys.outdim, self.outdim)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(self.indim)])

        rhs = self.linsys.RightHandSide(x)
        self.assertEqual(len(rhs), self.outdim)
        self.assertEqual(np.linalg.norm(rhs), 0.0)

        A = self.linsys.Operator(x)
        self.assertEqual(np.shape(A) [0], self.outdim)
        self.assertEqual(np.shape(A) [1], self.matdim)
        self.assertAlmostEqual(np.linalg.norm(A-self.mat), 0.0)

        para = clf.Parameters()
        para.Add('InputDimension', self.indim)
        para.Add('OutputDimension', self.matdim)
        para.Add('MaximumOrder', 4.0)
        para.Add('LocalRadius', 0.1)

        multiSet = clf.MultiIndexSet(para)
        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(self.indim)])

        func = clf.LocalFunction(multiSet, leg, center, para)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])

        fx = func.Evaluate(x, coeff);
        op = self.linsys.Operator(func, x, coeff);
        self.assertEqual(len(op), self.outdim)
        self.assertAlmostEqual(np.linalg.norm(A@fx-op), 0.0)

        jac = self.linsys.JacobianWRTCoefficients(func, x, coeff)
        self.assertEqual(np.shape(jac) [0], self.outdim)
        self.assertEqual(np.shape(jac) [1], len(coeff))
        jacFD = self.linsys.JacobianWRTCoefficientsFD(func, x, coeff)
        self.assertEqual(np.shape(jacFD) [0], self.outdim)
        self.assertEqual(np.shape(jacFD) [1], len(coeff))
        self.assertAlmostEqual(np.linalg.norm(jac-jacFD)/np.linalg.norm(jac), 0.0)

        weights = np.array([random.uniform(-1.0, 1.0) for i in range(self.outdim)])

        hess = self.linsys.HessianWRTCoefficients(func, x, coeff, weights)
        self.assertEqual(np.shape(hess) [0], len(coeff))
        self.assertEqual(np.shape(hess) [1], len(coeff))
        self.assertEqual(np.linalg.norm(hess), 0.0)
        hessFD = self.linsys.HessianWRTCoefficientsFD(func, x, coeff, weights)
        self.assertEqual(np.shape(hess) [0], len(coeff))
        self.assertEqual(np.shape(hess) [1], len(coeff))
        self.assertAlmostEqual(np.linalg.norm(hessFD, ord=np.inf), 0.0, 6)

    def test_square_identity(self):
        self.matdim = self.outdim
        self.mat = np.identity(self.outdim)
        self.linsys = clf.LinearModel(self.indim, self.outdim)

    def test_non_square_identity(self):
        self.matdim = 8
        self.mat = np.zeros((self.outdim, self.matdim))
        for i in range(min(self.outdim, self.matdim)):
            self.mat[i, i] = 1.0
        self.linsys = clf.LinearModel(self.indim, self.outdim, self.matdim)

    def test_random_matrix(self):
        self.matdim = 8
        self.mat = np.zeros((self.outdim, self.matdim))
        for i in range(self.outdim):
            for j in range(self.matdim):
                self.mat[i, j] = np.random.uniform(-1.0, 1.0)
        self.linsys = clf.LinearModel(self.indim, self.mat)

