"""! @brief Test the hypercube domain (clf::Hypercube) python interface"""

import unittest

import numpy as np
import scipy

import random

from TestPenaltyFunctions import *
import PyCoupledLocalFunctions as clf

class TestHypercube(unittest.TestCase):
    """! Test the hypercube domain (clf::Hypercube) python interface"""

    def test_unit_cube(self):
        """! Test the unit hypercube"""
        dim = 4

        para = clf.Parameters()
        para.Add("InputDimension", dim)

        dom0 = clf.Hypercube(dim)
        dom1 = clf.Hypercube(para)

        for i in range(dim):
            self.assertEqual(dom0.LeftBoundary(i), 0.0)
            self.assertEqual(dom1.LeftBoundary(i), 0.0)
            self.assertEqual(dom0.RightBoundary(i), 1.0)
            self.assertEqual(dom1.RightBoundary(i), 1.0)

        x = np.array([random.uniform(0.0, 1.0) for i in range(dim)])
        self.assertTrue(dom0.Inside(x))
        self.assertTrue(dom1.Inside(x))
        check = random.randint(0, dim-1)
        x[check] = 2.0
        self.assertFalse(dom0.Inside(x))
        self.assertFalse(dom1.Inside(x))
        x[check] = -2.44536
        self.assertFalse(dom0.Inside(x))
        self.assertFalse(dom1.Inside(x))
        x[check] = 0.44536
        self.assertTrue(dom0.Inside(x))
        self.assertTrue(dom1.Inside(x))

        y0 = dom0.MapToHypercube(x)
        y1 = dom1.MapToHypercube(x)
        cube = clf.Hypercube(-1.0, 1.0, dim);
        self.assertTrue(cube.Inside(y0))
        self.assertTrue(cube.Inside(y1))

        expected = 2.0*x-np.array([1.0]*dim)
        self.assertAlmostEqual(np.linalg.norm(y0-expected), 0.0)
        self.assertAlmostEqual(np.linalg.norm(y1-expected), 0.0)

        for i in range(10):
            samp0 = dom0.Sample()
            self.assertEqual(len(samp0), dim)
            self.assertTrue(dom0.Inside(samp0))
            samp1 = dom1.Sample()
            self.assertEqual(len(samp1), dim)
            self.assertTrue(dom1.Inside(samp1))
        
    def test_fixed_cube(self):
        """! Test the fixed hypercube"""
        left = -8.0
        right = 4.0
        dim = 4

        para = clf.Parameters()
        para.Add("InputDimension", dim)
        para.Add("LeftBoundary", left)
        para.Add("RightBoundary", right)

        dom0 = clf.Hypercube(left, right, dim)
        dom1 = clf.Hypercube(para)

        for i in range(dim):
            self.assertEqual(dom0.LeftBoundary(i), left)
            self.assertEqual(dom1.LeftBoundary(i), left)
            self.assertEqual(dom0.RightBoundary(i), right)
            self.assertEqual(dom1.RightBoundary(i), right)

        x = np.array([random.uniform(left, right) for i in range(dim)])
        self.assertTrue(dom0.Inside(x))
        self.assertTrue(dom1.Inside(x))
        check = random.randint(0, dim-1)
        x[check] = 10.0
        self.assertFalse(dom0.Inside(x))
        self.assertFalse(dom1.Inside(x))
        x[check] = -8.44536
        self.assertFalse(dom0.Inside(x))
        self.assertFalse(dom1.Inside(x))
        x[check] = 0.44536
        self.assertTrue(dom0.Inside(x))
        self.assertTrue(dom1.Inside(x))

        y0 = dom0.MapToHypercube(x)
        y1 = dom1.MapToHypercube(x)
        cube = clf.Hypercube(-1.0, 1.0, dim);
        self.assertTrue(cube.Inside(y0))
        self.assertTrue(cube.Inside(y1))

        center = np.array([(left+right)/2.0]*dim)
        delta = np.array([right-left])
        expected = 2.0*(x-center)/delta
        self.assertAlmostEqual(np.linalg.norm(y0-expected), 0.0)
        self.assertAlmostEqual(np.linalg.norm(y1-expected), 0.0)

        for i in range(10):
            samp0 = dom0.Sample()
            self.assertEqual(len(samp0), dim)
            self.assertTrue(dom0.Inside(samp0))
            samp1 = dom1.Sample()
            self.assertEqual(len(samp1), dim)
            self.assertTrue(dom1.Inside(samp1))

    def test_random_cube(self):
        """! Test the fixed hypercube"""
        left = -8.0
        right = 4.0
        dim = 4
        left = np.array([random.uniform(-3.0, -1.0) for i in range(dim)])
        right = np.array([random.uniform(0.0, 2.0) for i in range(dim)])

        dom = clf.Hypercube(left, right)

        for i in range(dim):
            self.assertEqual(dom.LeftBoundary(i), left[i])
            self.assertEqual(dom.RightBoundary(i), right[i])

        x = [random.uniform(left[i], right[i]) for i in range(dim)]
        self.assertTrue(dom.Inside(x))
        check = random.randint(0, dim-1)
        x[check] = 10.0
        self.assertFalse(dom.Inside(x))
        x[check] = -8.44536
        self.assertFalse(dom.Inside(x))
        x[check] = -0.44536
        self.assertTrue(dom.Inside(x))

        y = dom.MapToHypercube(x)
        cube = clf.Hypercube(-1.0, 1.0, dim);
        self.assertTrue(cube.Inside(y))

        center = (left+right)/2.0
        delta = [right-left]
        expected = 2.0*(x-center)/delta
        self.assertAlmostEqual(np.linalg.norm(y-expected), 0.0)

        for i in range(10):
            samp = dom.Sample()
            self.assertEqual(len(samp), dim)
            self.assertTrue(dom.Inside(samp))

    def test_superset(self):
        """! Test a domain with a superset defined"""
        left0 = -8.0
        right0 = 4.0
        left1 = -4.0
        right1 = 8.0
        dim = 4
        
        dom = clf.Hypercube(left0, right0, dim)
        sup = clf.Hypercube(left1, right1, dim)
        dom.SetSuperset(sup)

        check = random.randint(0, dim)
        x = np.array([0.0]*dim)
        x[check] = -5.0
        self.assertFalse(sup.Inside(x))
        self.assertFalse(dom.Inside(x))
        x[check] = 0.0
        self.assertTrue(sup.Inside(x))
        self.assertTrue(dom.Inside(x))

        for i in range(10):
            x = dom.Sample()
            self.assertTrue(sup.Inside(x))
            self.assertTrue(dom.Inside(x))


