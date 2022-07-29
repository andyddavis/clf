import unittest

import PyCoupledLocalFunctions as clf
import random

class TestOrthogonalPolynomials(unittest.TestCase):
    def test_Legendre(self):
        leg = clf.LegendrePolynomials() 

        x = random.uniform(-1, 1)

        self.assertAlmostEqual(leg.Evaluate(0, x), 1.0)
        self.assertAlmostEqual(leg.Evaluate(1, x), x)

        self.assertAlmostEqual(leg.Evaluate(0, 0.3), 1.0)
        self.assertAlmostEqual(leg.Evaluate(1, 0.3), 0.3)
        self.assertAlmostEqual(leg.Evaluate(2, 0.3), 0.5*(3.0*0.3**2.0-1.0))

        eval = leg.EvaluateAll(5, 0.325);
        print(eval)

        self.assertTrue(False)
