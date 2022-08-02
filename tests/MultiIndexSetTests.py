"""! @brief Test the multi index set (clf::MultiIndexSet) python interface"""

import unittest

import PyCoupledLocalFunctions as clf
import scipy

class TestMultIndexSet(unittest.TestCase):
    """! Test the multi index set (clf::MultiIndexSet) python interface"""
    def test_total_order(self):
        """! Test the total order construction of a multi index set"""
        dim = int(3)
        maxOrder = int(4)
    
        para = clf.Parameters()
        para.Add('InputDimension', dim)
        para.Add('MaximumOrder', maxOrder)
    
        multiSet = clf.MultiIndexSet(para)
    
        expectedNumIndices = 0;
        for i in range(maxOrder+1):
            expectedNumIndices += scipy.special.comb(dim+i-1, i);
        self.assertEqual(multiSet.NumIndices(), expectedNumIndices)

        for ind in multiSet.indices:
            self.assertLessEqual(ind.Order(), maxOrder)

        for i in range(dim):
            self.assertEqual(multiSet.MaxIndex(i), maxOrder)
