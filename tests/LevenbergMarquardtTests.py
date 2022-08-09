"""! @brief Test the Levenberg Marquardt algorithm (clf::LevenbergMarquardt) python interface"""

import unittest

import numpy as np
import scipy

import random

from TestPenaltyFunctions import *
import PyCoupledLocalFunctions as clf

class TestLevenbergMarquardt(unittest.TestCase):
    """! Test the Levenberg Marquardt (clf::LevenbergMarquardt) python interface"""

    def test_dense(self):
        para = clf.Parameters()

        func0 = DensePenaltyFunctionTest0()
        func1 = DensePenaltyFunctionTest1()
        cost = clf.DenseCostFunction([func0, func1])

        lm = clf.DenseLevenbergMarquardt(cost, para)
        self.assertEqual(lm.NumParameters(), cost.InputDimension())

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(lm.NumParameters())])
        result = lm.Minimize(beta)
        self.assertEqual(len(result), 4)
        self.assertTrue(result[0]==clf.OptimizationConvergence.CONVERGED or
                        result[0]==clf.OptimizationConvergence.CONVERGED_FUNCTION_SMALL or
                        result[0]==clf.OptimizationConvergence.CONVERGED_GRADIENT_SMALL)
        self.assertAlmostEqual(result[1], 0.0)
        expected = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(result[2]-expected), 0.0)
        self.assertAlmostEqual(np.linalg.norm(result[3]), 0.0)

    def test_sparse(self):
        para = clf.Parameters()

        func0 = SparsePenaltyFunctionTest0()
        func1 = SparsePenaltyFunctionTest1()
        cost = clf.SparseCostFunction([func0, func1])

        lm = clf.SparseLevenbergMarquardt(cost, para)
        self.assertEqual(lm.NumParameters(), cost.InputDimension())

        beta = np.array([random.uniform(-1.0, 1.0) for i in range(lm.NumParameters())])
