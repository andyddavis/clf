"""! @brief Test the residual of the weak form of the conservation law (clf::ConservationLawWeakFormResidual) python interface"""

import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestConservationLawWeakFormResidual(unittest.TestCase):
    """! Test the residual of the weak form of the conservation law (clf::ConservationLawWeakFormResidual) python interface"""

    def setUp(self):
        """! Set up the tests"""
        ## The input dimension
        self.indim = 4

    def tearDown(self):
        """! Run the test"""
        numPoints = 25
        para = clf.Parameters()
        para.Add('NumPoints', numPoints)

        radius = 1.5
        center = np.array([random.uniform(-1.0, 1.0) for i in range(self.indim)])
        domain = clf.Hypercube(center-np.array([radius]*self.indim), center+np.array([radius]*self.indim))
        # the local function that we are trying to fit
        maxOrder = 4
        multiSet = clf.MultiIndexSet(self.indim, maxOrder)
        basis = clf.LegendrePolynomials()
        func = clf.LocalFunction(multiSet, basis, domain, 1)
        
        # the feature vector that defines the test function
        vec = clf.FeatureVector(multiSet, basis)

        resid = clf.ConservationLawWeakFormResidual(func, self.system, vec, para)
        self.assertEqual(resid.indim, func.NumCoefficients())
        self.assertEqual(resid.outdim, vec.NumBasisFunctions())
        self.assertEqual(resid.NumBoundaryPoints(), numPoints)
        self.assertEqual(resid.NumPoints(), numPoints)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])

        expected = np.array([0.0]*resid.outdim)
        for i in range(resid.NumBoundaryPoints()):
            pt = resid.GetBoundaryPoint(i)
            expected += vec.Evaluate(pt.x)*np.dot(pt.normal, self.system.Flux(func, pt.x, coeff))/resid.NumBoundaryPoints()
        for i in range(resid.NumPoints()):
            pt = resid.GetPoint(i)
            expected -= ( vec.Derivative(pt.x, np.identity(self.indim))@self.system.Flux(func, pt.x, coeff) + vec.Evaluate(pt.x)*self.system.RightHandSide(pt.x) )/resid.NumPoints()

        computed = resid.Evaluate(coeff)
        self.assertEqual(len(computed), resid.outdim)
        self.assertAlmostEqual(np.linalg.norm(computed-expected), 0.0, places=10)

        jac = resid.Jacobian(coeff)
        self.assertEqual(np.shape(jac)[0], vec.NumBasisFunctions())
        self.assertEqual(np.shape(jac)[1], func.NumCoefficients())
        jacFD = resid.JacobianFD(coeff)
        self.assertEqual(np.shape(jacFD)[0], vec.NumBasisFunctions())
        self.assertEqual(np.shape(jacFD)[1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(jac-jacFD), 0.0, places=8)
        
        weights = np.array([random.uniform(-1.0, 1.0) for i in range(resid.outdim)])
        hess = resid.Hessian(coeff, weights)
        self.assertEqual(np.shape(hess)[0], func.NumCoefficients())
        self.assertEqual(np.shape(hess)[1], func.NumCoefficients())
        hessFD = resid.HessianFD(coeff, weights)
        self.assertEqual(np.shape(hessFD)[0], func.NumCoefficients())
        self.assertEqual(np.shape(hessFD)[1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(hess-hessFD), 0.0, places=8)

    def test_advection_equation(self):
        """! Test with the advection equation"""
        self.system = clf.AdvectionEquation(self.indim, 2.1);

    def test_Burgers_equation(self):
        """! Test with the Burgers equation"""
        self.system = clf.BurgersEquation(self.indim, 2.1);

