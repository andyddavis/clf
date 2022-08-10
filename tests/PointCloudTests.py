"""! @brief Test the point cloud (clf::PointCloud) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestPointCloud(unittest.TestCase):
   """! Test the point cloud (clf::PointCloud) python interface"""
   
   def test_add_points(self):
       dim = 3

       cloud = clf.PointCloud()

       x = np.array([random.uniform(-1.0, 1.0) for i in range(dim)])
       cloud.AddPoint(x)
       self.assertEqual(cloud.NumPoints(), 1)

       pt = clf.Point(np.array([random.uniform(-1.0, 1.0) for i in range(dim)]))
       cloud.AddPoint(pt)
       self.assertEqual(cloud.NumPoints(), 2)

       self.assertAlmostEqual(np.linalg.norm(cloud.Get(0).x-x), 0.0)
       self.assertAlmostEqual(np.linalg.norm(cloud.Get(1).x-pt.x), 0.0)

    
