"""! @brief Test the coupled location function (clf::CoupledLocalFunctions) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestCoupledLocalFunctions(unittest.TestCase):
    """! Test the coupled location function (clf::CoupledLocalFunctions) python interface"""
    def test_evaluation(self):
        """! Test evaluating the coupled local function"""
        dim = 5
        numPoints = 10
        
        cloud = clf.PointCloud()
        for i in range(numPoints):
            cloud.AddPoint(np.array([random.uniform(-1.0, 1.0) for i in range(dim)]))
            
        func = clf.CoupledLocalFunctions(cloud)
        self.assertEqual(func.NumLocalFunctions(), numPoints)
