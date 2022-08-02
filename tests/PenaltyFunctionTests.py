"""! @brief Test the penalty function (clf::PenaltyFunction) python interface"""

import unittest

import numpy as np
import scipy

import random

from TestPenaltyFunctions import *
import PyCoupledLocalFunctions as clf

class TestDensePenaltyFunction(unittest.TestCase):
    """! Test the dense penalty function (clf::DensePenaltyFunction) python interface"""
    def tearDown(self):
        """! Compare the derivative information to finite difference approximations"""
        
        self.assertEqual(self.func.indim, 3)
        self.assertEqual(self.func.outdim, self.outdim)

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(3)])

        jac = self.func.Jacobian(beta)
        self.assertEqual(np.shape(jac) [0], self.outdim)
        self.assertEqual(np.shape(jac) [1], 3)
        jacFD = self.func.JacobianFD(beta)
        self.assertEqual(np.shape(jacFD) [0], self.outdim)
        self.assertEqual(np.shape(jacFD) [1], 3)
        self.assertAlmostEqual(np.linalg.norm(jac-jacFD), 0.0)
        
        weights = np.array([random.uniform(-1.0, 1.0) for i in range(self.outdim)])
        hess = self.func.Hessian(beta, weights)
        self.assertEqual(np.shape(hess) [0], 3)
        self.assertEqual(np.shape(hess) [1], 3)
        hessFD = self.func.HessianFD(beta, weights)
        self.assertEqual(np.shape(hessFD) [0], 3)
        self.assertEqual(np.shape(hessFD) [1], 3)
        self.assertAlmostEqual(np.linalg.norm(hess-hessFD), 0.0)
    
    def test_dense_test0(self):
        """! Check TestPenaltyFunctions.DensePenaltyFunctionTest0"""

        ## The output dimension
        self.outdim = 2
        ## The penalty function
        self.func = DensePenaltyFunctionTest0()
        
        # evaluate the cost function 
        beta = np.array([random.uniform(-1.0, 1.0) for i in range(3)])
        fx = self.func.Evaluate(beta)
        self.assertEqual(len(fx), self.func.outdim);
        self.assertAlmostEqual(fx[0], beta[0])
        self.assertAlmostEqual(fx[1], beta[0]*(1.0-beta[2]))

    def test_dense_test1(self):
        """! Check TestPenaltyFunctions.DensePenaltyFunctionTest1"""
        
        ## The output dimension
        self.outdim = 6
        ## The penalty function
        self.func = DensePenaltyFunctionTest1()
        
        # evaluate the cost function 
        beta = np.array([random.uniform(-1.0, 1.0) for i in range(3)])
        fx = self.func.Evaluate(beta)
        self.assertEqual(len(fx), self.func.outdim);
        self.assertAlmostEqual(fx[0], 1.0-beta[1])
        self.assertAlmostEqual(fx[1], 1.0-beta[1]+beta[2])
        self.assertAlmostEqual(fx[2], beta[2])
        self.assertAlmostEqual(fx[3], beta[2]*(1.0-beta[1]))
        self.assertAlmostEqual(fx[4], beta[0]*beta[2])
        self.assertAlmostEqual(fx[5], beta[0]*beta[0]*beta[1])

class TestSparsePenaltyFunction(unittest.TestCase):
    """! Test the sparse penalty function (clf::SparsePenaltyFunction) python interface"""
    def tearDown(self):
        """! Compare the derivative information to finite difference approximations"""
                
        self.assertEqual(self.func.indim, 3)
        self.assertEqual(self.func.outdim, self.outdim)

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(3)])

        jac = self.func.Jacobian(beta)
        self.assertEqual(np.shape(jac) [0], self.outdim)
        self.assertEqual(np.shape(jac) [1], 3)
        jacFD = self.func.JacobianFD(beta)
        self.assertEqual(np.shape(jacFD) [0], self.outdim)
        self.assertEqual(np.shape(jacFD) [1], 3)
        self.assertAlmostEqual(scipy.sparse.linalg.norm(jac-jacFD), 0.0)
        
        weights = np.array([random.uniform(-1.0, 1.0) for i in range(self.outdim)])
        hess = self.func.Hessian(beta, weights)
        self.assertEqual(np.shape(hess) [0], 3)
        self.assertEqual(np.shape(hess) [1], 3)
        hessFD = self.func.HessianFD(beta, weights)
        self.assertEqual(np.shape(hessFD) [0], 3)
        self.assertEqual(np.shape(hessFD) [1], 3)
        self.assertAlmostEqual(scipy.sparse.linalg.norm(hess-hessFD), 0.0)

    def test_sparse_test0(self):
        """! Check TestPenaltyFunctions.SparsePenaltyFunctionTest0"""

        ## The output dimension
        self.outdim = 2
        ## The penalty function
        self.func = SparsePenaltyFunctionTest0()
        
        # evaluate the cost function 
        beta = np.array([random.uniform(-1.0, 1.0) for i in range(3)])
        fx = self.func.Evaluate(beta)
        self.assertEqual(len(fx), self.func.outdim);
        self.assertAlmostEqual(fx[0], beta[0])
        self.assertAlmostEqual(fx[1], beta[0]*(1.0-beta[2]))

    def test_sparse_test1(self):
        """! Check TestPenaltyFunctions.SparsePenaltyFunctionTest0"""

        ## The output dimension
        self.outdim = 6
        ## The penalty function
        self.func = SparsePenaltyFunctionTest1()
        
        # evaluate the cost function 
        beta = np.array([random.uniform(-1.0, 1.0) for i in range(3)])
        fx = self.func.Evaluate(beta)
        self.assertEqual(len(fx), self.func.outdim);
        self.assertAlmostEqual(fx[0], 1.0-beta[1])
        self.assertAlmostEqual(fx[1], 1.0-beta[1]+beta[2])
        self.assertAlmostEqual(fx[2], beta[2])
        self.assertAlmostEqual(fx[3], beta[2]*(1.0-beta[1]))
        self.assertAlmostEqual(fx[4], beta[0]*beta[2])
        self.assertAlmostEqual(fx[5], beta[0]*beta[0]*beta[1])

    
