"""! @brief Test the cost function (clf::CostFunction) python interface"""

import unittest

import numpy as np
import scipy

import random

from TestPenaltyFunctions import *
import PyCoupledLocalFunctions as clf

class TestCostFunction(unittest.TestCase):
    """! Test the cost function (clf::CostFunction) python interface"""

    def tearDown(self):
        """! Check the evaluation and derivative sizes"""
        
        self.assertEqual(self.cost.InputDimension(), self.func0.indim)
        self.assertEqual(self.cost.InputDimension(), self.func1.indim)
        self.assertEqual(self.cost.numPenaltyFunctions, 2)
        self.assertEqual(self.cost.numTerms, self.func0.outdim+self.func1.outdim)

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(self.cost.InputDimension())])

        fx = self.cost.Evaluate(beta)
        self.assertAlmostEqual(fx[0], beta[0])
        self.assertAlmostEqual(fx[1], beta[0]*(1.0-beta[2]))
        self.assertAlmostEqual(fx[2], 1.0-beta[1])
        self.assertAlmostEqual(fx[3], 1.0-beta[1]+beta[2])
        self.assertAlmostEqual(fx[4], beta[2])
        self.assertAlmostEqual(fx[5], beta[2]*(1.0-beta[1]))
        self.assertAlmostEqual(fx[6], beta[0]*beta[2])
        self.assertAlmostEqual(fx[7], beta[0]*beta[0]*beta[1])

        grad = self.cost.Gradient(beta);
        self.assertEqual(len(grad), self.func0.indim)
        self.assertEqual(len(grad), self.func1.indim)
        
        hess = self.cost.Hessian(beta)
        self.assertEqual(np.shape(hess) [0], self.func0.indim)
        self.assertEqual(np.shape(hess) [0], self.func1.indim)
        self.assertEqual(np.shape(hess) [1], self.func0.indim)
        self.assertEqual(np.shape(hess) [1], self.func1.indim)

    def test_dense(self):
        """! Test the dense cost function (clf::DenseCostFunction) python interface"""

        ## The first penalty function
        self.func0 = DensePenaltyFunctionTest0()
        ## The second penalty function
        self.func1 = DensePenaltyFunctionTest1()
        ## The cost function
        self.cost = clf.DenseCostFunction([self.func0, self.func1])

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(self.cost.InputDimension())])
        jac = self.cost.Jacobian(beta)
        self.assertEqual(np.shape(jac) [1], self.func0.indim)
        self.assertEqual(np.shape(jac) [1], self.func1.indim)
        self.assertEqual(np.shape(jac) [0], self.func0.outdim+self.func1.outdim)
        self.assertAlmostEqual(np.linalg.norm(jac[0:self.func0.outdim, 0:self.func0.indim] - self.func0.Jacobian(beta)), 0.0)
        self.assertAlmostEqual(np.linalg.norm(jac[self.func0.outdim:self.func0.outdim+self.func1.outdim, 0:self.func1.indim] - self.func1.Jacobian(beta)), 0.0)

    def test_sparse(self):
        """! Test the sparse cost function (clf::SparseCostFunction) python interface"""

        ## The first penalty function
        self.func0 = SparsePenaltyFunctionTest0()
        ## The second penalty function
        self.func1 = SparsePenaltyFunctionTest1()
        ## The cost function
        self.cost = clf.SparseCostFunction([self.func0, self.func1])

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(self.cost.InputDimension())])
        jac = self.cost.Jacobian(beta)
        self.assertEqual(np.shape(jac) [1], self.func0.indim)
        self.assertEqual(np.shape(jac) [1], self.func1.indim)
        self.assertEqual(np.shape(jac) [0], self.func0.outdim+self.func1.outdim)
        self.assertAlmostEqual(scipy.sparse.linalg.norm(jac[0:self.func0.outdim, 0:self.func0.indim] - self.func0.Jacobian(beta)), 0.0)
        self.assertAlmostEqual(scipy.sparse.linalg.norm(jac[self.func0.outdim:self.func0.outdim+self.func1.outdim, 0:self.func1.indim] - self.func1.Jacobian(beta)), 0.0)
