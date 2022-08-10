"""! @brief Test the system of equations (clf::SystemOfEquations) python interface"""

import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestSystemOfEquations(unittest.TestCase):
    """! @Test the system of equations (clf::SystemOfEquations) python interface"""

    def test_default_implementation(self):
        """! Test the default implementation"""
        indim = 2
        outdim = 3
        
        sys = clf.SystemOfEquations(indim, outdim)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])

        rhs = sys.RightHandSide(x)
        self.assertEqual(len(rhs), outdim)
        self.assertEqual(np.linalg.norm(rhs), 0.0)

