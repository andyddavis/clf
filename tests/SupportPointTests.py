import unittest

import numpy as np

import CoupledLocalFunctions as clf

class TestSupportPoint(unittest.TestCase):
    def test_create(self):
        x = np.array([1.0, 4.0])
        d1 = dict()
        d1["Type"] = "TotalOrderPolynomials"
        d = dict()
        d["BasisFunctions"] = d1
        point = clf.SupportPoint(x, d)
