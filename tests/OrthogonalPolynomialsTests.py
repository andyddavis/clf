"""! @brief Test the orthogonal polynomials (clf::OrthogonalPolynomials) python interface"""

import unittest

import PyCoupledLocalFunctions as clf
import random

class TestOrthogonalPolynomials(unittest.TestCase):
    """! Test the orthogonal polynomials (clf::OrthogonalPolynomials) python interface"""
    
    def test_Legendre(self):
        """! Test the Legendre polynomials"""
        leg = clf.LegendrePolynomials() 

        x = random.uniform(-1, 1)

        self.assertAlmostEqual(leg.Evaluate(0, x), 1.0)
        self.assertAlmostEqual(leg.Evaluate(1, x), x)
        self.assertAlmostEqual(leg.Evaluate(2, x), 1.5*x*x-0.5)

        eval = leg.EvaluateAll(5, 0.325);
        self.assertEqual(len(eval), 6)
        self.assertAlmostEqual(eval[0], 1.0)
        self.assertAlmostEqual(eval[1], 0.325)
        self.assertAlmostEqual(eval[5], 0.3375579333496094)

        self.assertAlmostEqual(-0.05346106275520913, leg.Evaluate(20, -0.845))
        self.assertAlmostEqual(-0.1119514835092105, leg.Evaluate(50, 0.1264))
        self.assertAlmostEqual(-0.001892916076323403, leg.Evaluate(200, -0.3598))
        self.assertAlmostEqual(0.01954143166718206, leg.Evaluate(1000, 0.4587))

        # evaluate derivatives at a known location
        x = 0.23;
        derivs = leg.EvaluateAllDerivatives(4, x, 3);

        # first derivatives at a known location
        self.assertAlmostEqual(0.0, leg.EvaluateDerivative(0, x, 1));
        self.assertAlmostEqual(0.0, derivs[0, 0]);
        self.assertAlmostEqual(1.0, leg.EvaluateDerivative(1, x, 1));
        self.assertAlmostEqual(1.0, derivs[1, 0]);
        self.assertAlmostEqual(3.0*x, leg.EvaluateDerivative(2, x, 1));
        self.assertAlmostEqual(3.0*x, derivs[2, 0]);
        self.assertAlmostEqual(7.5*x*x - 1.5, leg.EvaluateDerivative(3, x, 1));
        self.assertAlmostEqual(7.5*x*x - 1.5, derivs[3, 0]);
        self.assertAlmostEqual(17.5*x*x*x - 7.5*x, leg.EvaluateDerivative(4, x, 1));
        self.assertAlmostEqual(17.5*x*x*x - 7.5*x, derivs[4, 0]);
        
        # second derivatives
        self.assertAlmostEqual(0.0, leg.EvaluateDerivative(0, x, 2))
        self.assertAlmostEqual(0.0, derivs[0, 1])
        self.assertAlmostEqual(0.0, leg.EvaluateDerivative(1, x, 2))
        self.assertAlmostEqual(0.0, derivs[1, 1])
        self.assertAlmostEqual(3.0, leg.EvaluateDerivative(2, x, 2))
        self.assertAlmostEqual(3.0, derivs[2, 1])
        self.assertAlmostEqual(15.0*x, leg.EvaluateDerivative(3, x, 2))
        self.assertAlmostEqual(15.0*x, derivs[3, 1])
        self.assertAlmostEqual(52.5*x*x - 7.5, leg.EvaluateDerivative(4, x, 2))
        self.assertAlmostEqual(52.5*x*x - 7.5, derivs[4, 1])

        # third derivatives
        self.assertAlmostEqual(0.0, leg.EvaluateDerivative(0, x, 3))
        self.assertAlmostEqual(0.0, derivs[0, 2])
        self.assertAlmostEqual(0.0, leg.EvaluateDerivative(1, x, 3))
        self.assertAlmostEqual(0.0, derivs[1, 2])
        self.assertAlmostEqual(0.0, leg.EvaluateDerivative(2, x, 3))
        self.assertAlmostEqual(0.0, derivs[2, 2])
        self.assertAlmostEqual(15.0, leg.EvaluateDerivative(3, x, 3))
        self.assertAlmostEqual(15.0, derivs[3, 2])
        self.assertAlmostEqual(105.0*x, leg.EvaluateDerivative(4, x, 3))
        self.assertAlmostEqual(105.0*x, derivs[4, 2])
