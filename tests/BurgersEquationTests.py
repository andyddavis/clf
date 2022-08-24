"""! @brief Test the advection equation (clf::AdvectionEquation) python interface"""

import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestAdvectionEquation(unittest.TestCase):
    """! Test the advection equation (clf::AdvectionEquation) python interface"""

    def setUp(self):
        ## The input dimension
        self.indim = 4

    def tearDown(self):
        self.assertEqual(self.system.indim, self.indim)
        self.assertEqual(self.system.outdim, 1)

        maxOrder = 4
        multiSet = clf.MultiIndexSet(self.indim, maxOrder)

        radius = 1.5
        center = np.array([random.uniform(-1.0, 1.0) for i in range(self.indim)])
        domain = clf.Hypercube(center-np.array([radius]*self.indim), center+np.array([radius]*self.indim))
        
        x = center + np.array([random.uniform(-radius, radius) for i in range(self.indim)])

        basis = clf.LegendrePolynomials()
        func = clf.LocalFunction(multiSet, basis, domain, 1)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])

        flux = self.system.Flux(func, x, coeff)
        self.assertEqual(len(flux), self.indim)
        fx = func.Evaluate(x, coeff) [0]
        expectedFlux = [0.0]*self.indim
        expectedFlux[0] = fx
        expectedFlux[1:] = 0.5*self.velocity*fx*fx
        self.assertEqual(len(expectedFlux), self.indim)
        self.assertAlmostEqual(np.linalg.norm(expectedFlux-flux), 0.0, places=10)

        fluxJac = self.system.Flux_JacobianWRTCoefficients(func, x, coeff)
        self.assertEqual(np.shape(fluxJac) [0], self.indim)
        self.assertEqual(np.shape(fluxJac) [1], func.NumCoefficients())
        fluxJacFD = self.system.Flux_JacobianWRTCoefficientsFD(func, x, coeff)
        self.assertEqual(np.shape(fluxJacFD) [0], self.indim)
        self.assertEqual(np.shape(fluxJacFD) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(fluxJac-fluxJacFD), 0.0, places=10)

        weights = np.array([random.uniform(-1.0, 1.0) for i in range(self.indim)])
        fluxHess = self.system.Flux_HessianWRTCoefficients(func, x, coeff, weights)
        self.assertEqual(np.shape(fluxHess) [0], func.NumCoefficients())
        self.assertEqual(np.shape(fluxHess) [1], func.NumCoefficients())
        fluxHessFD = self.system.Flux_HessianWRTCoefficientsFD(func, x, coeff, weights)
        self.assertEqual(np.shape(fluxHessFD) [0], func.NumCoefficients())
        self.assertEqual(np.shape(fluxHessFD) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(fluxHess-fluxHessFD), 0.0, places=10)

        div = self.system.FluxDivergence(func, x, coeff)
        divFD = self.system.FluxDivergenceFD(func, x, coeff)
        self.assertAlmostEqual(div, divFD, places=10)

        divGrad = self.system.FluxDivergence_GradientWRTCoefficients(func, x, coeff)
        self.assertEqual(len(divGrad), func.NumCoefficients())
        divGradFD = self.system.FluxDivergence_GradientWRTCoefficientsFD(func, x, coeff)
        self.assertEqual(len(divGradFD), func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(divGrad-divGradFD), 0.0, places=10)

        divHess = self.system.FluxDivergence_HessianWRTCoefficients(func, x, coeff)
        self.assertEqual(np.shape(divHess) [0], func.NumCoefficients())
        self.assertEqual(np.shape(divHess) [1], func.NumCoefficients())
        divHessFD = self.system.FluxDivergence_HessianWRTCoefficientsFD(func, x, coeff)
        self.assertEqual(np.shape(divHessFD) [0], func.NumCoefficients())
        self.assertEqual(np.shape(divHessFD) [1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(divHess-divHessFD), 0.0, places=10)

        op = self.system.Operator(func, x, coeff)
        self.assertEqual(len(op), 1)
        self.assertAlmostEqual(op[0], div, places=10)

        jac = self.system.JacobianWRTCoefficients(func, x, coeff)
        self.assertEqual(np.shape(jac)[0], 1)
        self.assertEqual(np.shape(jac)[1], func.NumCoefficients())
        jacFD = self.system.JacobianWRTCoefficientsFD(func, x, coeff)
        self.assertEqual(np.shape(jacFD)[0], 1)
        self.assertEqual(np.shape(jacFD)[1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(jac-jacFD), 0.0, places=10)

        weights = np.array([random.uniform(-1.0, 1.0)])
        hess = self.system.HessianWRTCoefficients(func, x, coeff, weights)
        self.assertEqual(np.shape(hess)[0], func.NumCoefficients())
        self.assertEqual(np.shape(hess)[1], func.NumCoefficients())
        hessFD = self.system.HessianWRTCoefficientsFD(func, x, coeff, weights)
        self.assertEqual(np.shape(hessFD)[0], func.NumCoefficients())
        self.assertEqual(np.shape(hessFD)[1], func.NumCoefficients())
        self.assertAlmostEqual(np.linalg.norm(hess-hessFD), 0.0, places=10)

    def test_constant_velocity(self):
        """! Test the advection equation with a scalar velocity"""
        ## The velocity that defines the advection equation
        self.velocity = np.array([2.5]*(self.indim-1))
        
        ## The advection equation
        self.system = clf.BurgersEquation(self.indim, 2.5)

    def test_non_constant_velocity(self):
        """! Test the advection equation with a vector velocity"""
        ## The velocity that defines the advection equation
        self.velocity = np.array([random.uniform(-1.0, 1.0) for i in range(self.indim-1)])
        
        ## The advection equation
        self.system = clf.BurgersEquation(self.velocity)

