"""! @brief Test the hypercube domain (clf::Hypercube) python interface"""

import unittest

import numpy as np
import scipy

import random

from TestPenaltyFunctions import *
import PyCoupledLocalFunctions as clf

class TestHypercube(unittest.TestCase):
    """! Test the hypercube domain (clf::Hypercube) python interface"""
    def setUp(self):
        """! Set up the test environment"""

        ## The domain dimension
        self.dim = 4

        ## The domain parameters
        self.para = clf.Parameters()

    def CheckDomain(self, dom, left, right):
        """! Check that the clf::Hypercube is correct 
        @param[in] dom The domain we are checking
        @param[in] left The left boundary for each dimension
        @param[in] right The right boundary for each dimension
        """
        self.assertEqual(dom.dim, self.dim)

        for i in range(self.dim):
            self.assertAlmostEqual(dom.LeftBoundary(i), left[i], places=12)
            self.assertAlmostEqual(dom.RightBoundary(i), right[i], places=12)

        x = np.array([random.uniform(left[i], right[i]) for i in range(self.dim)])
        self.assertTrue(dom.Inside(x))
        check = random.randint(0, self.dim-1)
        x[check] = right[check]+10.0
        self.assertFalse(dom.Inside(x))
        x[check] = left[check]-3.5403
        self.assertFalse(dom.Inside(x))
        x[check] = (left[check]+right[check])/2.0
        self.assertTrue(dom.Inside(x))

        y = dom.MapToHypercube(x)
        cube = clf.Hypercube(-1.0, 1.0, self.dim)
        self.assertTrue(cube.Inside(y))

        center = (left+right)/2.0
        delta = right-left
        expected = 2.0*(x-center)/delta
        self.assertAlmostEqual(np.linalg.norm(y-expected), 0.0, places=10)

        for i in range(10):
            samp = dom.Sample()
            self.assertEqual(len(samp), self.dim)
            self.assertTrue(dom.Inside(samp))

    def CheckPeriodicDomain(self, dom, left, right, periodic):
        """! Check that the clf::Hypercube is correct 
        @param[in] dom The domain we are checking
        @param[in] left The left boundary for each dimension
        @param[in] right The right boundary for each dimension
        @param[in] periodic Which boundaries are periodic?
        """
        self.assertEqual(dom.dim, self.dim)

        for i in range(self.dim):
            self.assertAlmostEqual(dom.LeftBoundary(i), left[i], places=12)
            self.assertAlmostEqual(dom.RightBoundary(i), right[i], places=12)

        # a random point that does not wrap around
        x1 = np.array([random.uniform(left[i]+(right[i]-left[i])/4.0, right[i]-(right[i]-left[i])/4.0) for i in range(self.dim)])
        x2 = np.array([random.uniform(left[i]+(right[i]-left[i])/4.0, right[i]-(right[i]-left[i])/4.0) for i in range(self.dim)])
        self.assertTrue(dom.Inside(x1))
        self.assertTrue(dom.Inside(x2))
        dist = dom.Distance(x1, x2)
        self.assertAlmostEqual(dist, np.linalg.norm(x1-x2), places=10)

        # a random point that does wrap around
        x1 = np.array([random.uniform(left[i], left[i]+(right[i]-left[i])/4.0) for i in range(self.dim)])
        x2 = np.array([random.uniform(right[i]-(right[i]-left[i])/4.0, right[i]) for i in range(self.dim)])
        self.assertTrue(dom.Inside(x1))
        self.assertTrue(dom.Inside(x2))
        dist = dom.Distance(x1, x2)
        self.assertTrue(dist<np.linalg.norm(x1-x2))
        expected = 0.0
        for i in range(self.dim):
            if periodic[i]:
                mn = min(x1[i], x2[i])
                mx = max(x1[i], x2[i])
                diff = min(mx-mn, right[i]-mx+mn-left[i])
                expected += diff*diff
            else:
                diff = x1[i]-x2[i]
                expected += diff*diff
        self.assertAlmostEqual(dist, np.sqrt(expected), places=10)

        length = right[i]-left[i]
        check = random.randint(0, self.dim-1)
        while not periodic[check]:
            check = random.randint(0, self.dim-1)
        x1[check] = left[check]-length/5.0
        self.assertTrue(dom.Inside(x1))
        y = dom.MapPeriodic(x1)
        checkDom = clf.Hypercube(left, right)
        self.assertFalse(checkDom.Inside(x1))
        self.assertTrue(checkDom.Inside(y))
        
    def test_unit_cube(self):
        """! Test the unit hypercube"""
        dom = clf.Hypercube(self.dim)
        self.CheckDomain(dom, np.array([0.0]*self.dim), np.array([1.0]*self.dim))

        self.para.Add("InputDimension", self.dim)
        dom = clf.Hypercube(self.para)
        self.CheckDomain(dom, np.array([0.0]*self.dim), np.array([1.0]*self.dim))

    def test_unit_cube_periodic(self):
        """! Test the unit hypercube with periodic boundaries"""
        periodic = [True]*self.dim
        dom = clf.Hypercube(periodic)
        self.CheckPeriodicDomain(dom, np.array([0.0]*self.dim), np.array([1.0]*self.dim), periodic)

        periodic = [i==self.dim-1 for i in range(self.dim)]
        dom = clf.Hypercube(periodic, self.para)
        self.CheckPeriodicDomain(dom, np.array([0.0]*self.dim), np.array([1.0]*self.dim), periodic)
        
    def test_fixed_cube(self):
        """! Test the fixed hypercube"""
        left = -8.0
        right = 4.0

        dom = clf.Hypercube(left, right, self.dim)
        self.CheckDomain(dom, np.array([left]*self.dim), np.array([right]*self.dim))

        self.para.Add("InputDimension", self.dim)
        self.para.Add("LeftBoundary", left)
        self.para.Add("RightBoundary", right)
        dom = clf.Hypercube(self.para)
        self.CheckDomain(dom, np.array([left]*self.dim), np.array([right]*self.dim))

    def test_fixed_cube_periodic(self):
        """! Test the fixed hypercube with periodic boundaries"""
        left = -8.0
        right = 4.0

        periodic = [True]*self.dim
        dom = clf.Hypercube(periodic, left, right)
        self.CheckPeriodicDomain(dom, np.array([left]*self.dim), np.array([right]*self.dim), periodic)

        periodic[0] = False
        self.para.Add("LeftBoundary", left)
        self.para.Add("RightBoundary", right)
        dom = clf.Hypercube(periodic, self.para)
        self.CheckPeriodicDomain(dom, np.array([left]*self.dim), np.array([right]*self.dim), periodic)

    def test_random_cube(self):
        """! Test the fixed hypercube"""
        left = np.array([random.uniform(-3.0, -1.0) for i in range(self.dim)])
        right = np.array([random.uniform(0.0, 2.0) for i in range(self.dim)])

        dom = clf.Hypercube(left, right)
        self.CheckDomain(dom, left, right)

    def test_random_cube_periodic(self):
        """! Test the fixed hypercube with periodic boundaries"""
        left = np.array([random.uniform(-3.0, -1.0) for i in range(self.dim)])
        right = np.array([random.uniform(0.0, 2.0) for i in range(self.dim)])

        periodic = [False]*self.dim
        periodic[0] = True
        dom = clf.Hypercube(left, right, periodic)
        self.CheckPeriodicDomain(dom, left, right, periodic)

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

        check = random.randint(0, dim-1)
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


    def test_periodic_superset(self):
        globalDomain = clf.Hypercube(np.array([0.0, 0.0]), np.array([1.0, 1.0]), [False, True])
        
        center = np.array([0.65218484, 0.20443575])
        delta = np.array([1.0/(2.0*np.sqrt(10))]*2)
        domain = clf.Hypercube(center-delta, center+delta)
        domain.SetSuperset(globalDomain)

        point = np.array([1.0, 1.0])
        y = domain.MapPeriodic(point)
        expected = np.array([1.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(y-expected), 0.0, places=10)
