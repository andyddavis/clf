"""! @brief Test the point (clf::Point) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestPoint(unittest.TestCase):
   """! Test the point (clf::Point) python interface"""
   def test_set_point(self):
      """! Test adding and getting a parameter"""
      dim = 5
      x = np.array([random.uniform(-1.0, 1.0) for i in range(dim)])

      point = clf.Point(x)
      self.assertAlmostEqual(np.linalg.norm(point.x-x), 0.0)
