import unittest

import numpy as np

import CoupledLocalFunctions as clf

class TestBasisFunctions(unittest.TestCase):
    def test_total_order_polynomials(self):
        d = dict()
        d["InputDimension"] = 2
        d["Order"] = 3
        basis = clf.PolynomialBasis.TotalOrderBasis(d)

        self.assertEqual(basis.NumBasisFunctions(), 10)

    def test_total_order_sincos(self):
        d = dict()
        d["InputDimension"] = 2
        d["Order"] = 3
        basis = clf.SinCosBasis.TotalOrderBasis(d)

        self.assertEqual(basis.NumBasisFunctions(), 28)
