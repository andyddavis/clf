import unittest

import numpy as np

import CoupledLocalFunctions as clf

class Observations(clf.Model):
    def __init__(self, options):
        clf.Model.__init__(self, options)

    def RightHandSideComponentImpl(self, x, outind):
        if outind==0:
            return np.sin(np.pi*x[0])*np.cos(2.0*np.pi*x[2]) + np.cos(np.pi*x[2])
        elif outind==1:
            return x[1]*x[0] + x[0] + 1.0
        else:
            return 0.0

class TestLocalFunction(unittest.TestCase):
    def test_create(self):
        indim = 3
        outdim = 2

        # create the observation model
        modelOptions = dict()
        modelOptions["InputDimension"] = indim
        modelOptions["OutputDimension"] = outdim
        model = Observations(modelOptions)

        # options for the basis
        orderPoly = 5
        orderSinCos = 2
        suppOptions = dict()
        suppOptions["NumNeighbors"] = 75
        suppOptions["BasisFunctions"] = "Basis1, Basis2"
        suppOptions["Basis1.Type"] = "TotalOrderSinCos"
        suppOptions["Basis1.Order"] = orderSinCos
        suppOptions["Basis1.LocalBasis"] = False
        suppOptions["Basis2.Type"] = "TotalOrderPolynomials"
        suppOptions["Basis2.Order"] = orderPoly

        # create the support points
        supportPoints = [None]*150
        for i in range(len(supportPoints)):
            supportPoints[i] = clf.SupportPoint(0.1*np.random.rand(indim), model, suppOptions)

        # create the support point cloud
        cloudOptions = dict()
        cloud = clf.SupportPointCloud(supportPoints, cloudOptions)

        # create the local function
        funcOptions = dict()
        func = clf.LocalFunction(cloud, funcOptions)

        # the cost of the optimial coefficients
        cost = func.CoefficientCost()
        self.assertAlmostEqual(cost, float(0.0), places=8)

        for pnt in supportPoints:
            eval = pnt.EvaluateLocalFunction(pnt.x)
            expected = np.array([np.sin(np.pi*pnt.x[0])*np.cos(2.0*np.pi*pnt.x[2]) + np.cos(np.pi*pnt.x[2]), pnt.x[1]*pnt.x[0] + pnt.x[0] + 1.0])
            self.assertAlmostEqual(np.linalg.norm(eval-expected), 0.0, delta=10.0*np.sqrt(cost))

        for i in range(10):
            x = 0.1*np.random.rand(indim)
            eval = func.Evaluate(x)
            expected = np.array([np.sin(np.pi*x[0])*np.cos(2.0*np.pi*x[2]) + np.cos(np.pi*x[2]), x[1]*x[0] + x[0] + 1.0])
            self.assertAlmostEqual(np.linalg.norm(eval-expected), 0.0, delta=10.0*np.sqrt(cost))
