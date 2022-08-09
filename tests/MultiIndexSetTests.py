"""! @brief Test the multi index set (clf::MultiIndexSet) python interface"""

import unittest

import PyCoupledLocalFunctions as clf
import scipy

class TestTotalOrderMultIndexSet(unittest.TestCase):
    """! Test the multi index set (clf::MultiIndexSet) python interface"""
    def setUp(self):
        """! Set up the dimension and maximum order"""
        ## The input dimension
        self.dim = int(3)
        ## The output dimension
        self.maxOrder = int(4)

    def tearDown(self):
        """! Test the multi-index set"""
        expectedNumIndices = 0;
        for i in range(self.maxOrder+1):
            expectedNumIndices += scipy.special.comb(self.dim+i-1, i);
        self.assertEqual(self.multiSet.NumIndices(), expectedNumIndices)

        for ind in self.multiSet.indices:
            self.assertLessEqual(ind.Order(), self.maxOrder)

        for i in range(self.dim):
            self.assertEqual(self.multiSet.MaxIndex(i), self.maxOrder)
        
    def test_direct_construction(self):
        """! Test the total order construction of a multi index set using direction constructors"""
        ## The multi-index set
        self.multiSet = clf.MultiIndexSet(self.dim, self.maxOrder)

    def test_parameter_construction(self):
        """! Test the total order construction of a multi index set using a parameter type"""
        para = clf.Parameters()
        para.Add('InputDimension', self.dim)
        para.Add('MaximumOrder', self.maxOrder)

        ## The multi-index set
        self.multiSet = clf.MultiIndexSet(para)

