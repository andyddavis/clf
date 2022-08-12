"""! @brief Test the point cloud (clf::PointCloud) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestPointCloud(unittest.TestCase):
   """! Test the point cloud (clf::PointCloud) python interface"""
   
   def test_add_points(self):
      """! Test adding points to the cloud"""
      dim = 3
      
      cloud = clf.PointCloud()
      self.assertEqual(cloud.NumPoints(), 0)
      
      firstPt = clf.Point(np.array([random.uniform(-1.0, 1.0) for i in range(dim)]))
      
      pt1 = clf.Point(np.array([random.uniform(-1.0, 1.0) for i in range(dim)]))
      cloud.AddPoint(pt1)
      self.assertEqual(cloud.NumPoints(), 1)
      
      cloud.AddPoint(pt1)
      self.assertEqual(cloud.NumPoints(), 1)
      
      pt2 = clf.Point(np.array([random.uniform(-1.0, 1.0) for i in range(dim)]))
      cloud.AddPoint(pt2)
      self.assertEqual(cloud.NumPoints(), 2)
      
      cloud.AddPoint(pt1)
      cloud.AddPoint(pt2)
      self.assertEqual(cloud.NumPoints(), 2)
      
      cloud.AddPoint(firstPt)
      self.assertEqual(cloud.NumPoints(), 3)
      
      cloud.AddPoint(firstPt)
      cloud.AddPoint(pt1)
      cloud.AddPoint(pt2)
      self.assertEqual(cloud.NumPoints(), 3)
      
      def SortKey(p):
         return p.id

      pts = [pt1, pt2, firstPt]
      pts.sort(key=SortKey)
       
      for i in range(len(pts)):
         self.assertAlmostEqual(np.linalg.norm(cloud.Get(i).x-pts[i].x), 0.0)

   def test_domain(self):
      """! Test adding points to the cloud with a domain defined"""
      dim = 5 # the dimension 
      n = 100 # the number of points
      
      domain = clf.Hypercube(np.array([0.0]*dim), np.array([1.0]*dim))
      cloud = clf.PointCloud(domain)
      self.assertEqual(cloud.NumPoints(), 0)

      cloud.AddPoint()
      self.assertEqual(cloud.NumPoints(), 1)
      cloud.AddPoints(n-1)
      self.assertEqual(cloud.NumPoints(), n)

      for i in range(n):
         pi = cloud.Get(i)
         for j in range(n):
            pj = cloud.Get(j)
            expected = np.linalg.norm(pi.x-pj.x)
            self.assertAlmostEqual(cloud.Distance(i, j), expected)

      numNeighs = random.randint(0, n-1)
      
      for i in range(cloud.NumPoints()):
         neighbors = cloud.NearestNeighbors(i, numNeighs)
         self.assertEqual(len(neighbors), numNeighs)
         self.assertEqual(neighbors[0], i)
         for k in range(1, numNeighs):
            self.assertNotEqual(neighbors[k], i)
            self.assertTrue(cloud.Distance(i, neighbors[k-1])<cloud.Distance(i, neighbors[k]))
         for j in range(cloud.NumPoints()):
            if j==i or j in neighbors:
               continue
            self.assertTrue(cloud.Distance(i, j)>cloud.Distance(i, neighbors[-1]))

      cloud.AddPoints(n)
      self.assertEqual(cloud.NumPoints(), 2*n)

      for i in range(cloud.NumPoints()):
         neighbors = cloud.NearestNeighbors(i, numNeighs)
         self.assertEqual(len(neighbors), numNeighs)
         self.assertEqual(neighbors[0], i)
         for k in range(1, numNeighs):
            self.assertNotEqual(neighbors[k], i)
            self.assertTrue(cloud.Distance(i, neighbors[k-1])<cloud.Distance(i, neighbors[k]))
         for j in range(cloud.NumPoints()):
            if j==i or j in neighbors:
               continue
            self.assertTrue(cloud.Distance(i, j)>cloud.Distance(i, neighbors[-1]))
            
      x = [random.uniform(-1.0, 1.0) for i in range(dim)]
      [nearestInd, nearestDist] = cloud.ClosestPoint(x)
      for j in range(cloud.NumPoints()):
         if j==nearestInd:
            continue
         self.assertTrue(domain.Distance(x, cloud.Get(j).x)>nearestDist)
