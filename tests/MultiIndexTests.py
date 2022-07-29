import unittest

import PyCoupledLocalFunctions as clf
import random

class TestMultIndex(unittest.TestCase):
    def test_basic(self):
        dim = int(5)
        alpha = [None]*dim
        order = 0
        for i in range(dim): 
            alpha[i] = random.randint(0, 13)
            order += alpha[i]
        
        ind = clf.MultiIndex(alpha)
        self.assertEqual(ind.Dimension(), dim)
        self.assertEqual(ind.Order(), order)
      
