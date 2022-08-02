"""! @brief Test the multi-index (clf::MultiIndex) python interface"""

import unittest

import PyCoupledLocalFunctions as clf
import random

class TestMultIndex(unittest.TestCase):
    """! Test the multi-index (clf::MultiIndex) python interface"""
    def test_construction(self):
        """! Test the multi-index construction"""
        dim = int(5)
        alpha = [None]*dim
        order = 0
        for i in range(dim): 
            alpha[i] = random.randint(0, 13)
            order += alpha[i]
        
        ind = clf.MultiIndex(alpha)
        self.assertEqual(ind.Dimension(), dim)
        self.assertEqual(ind.Order(), order)
      
