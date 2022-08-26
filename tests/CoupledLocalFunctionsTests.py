"""! @brief Test the coupled location function (clf::CoupledLocalFunctions) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestCoupledLocalFunctions(unittest.TestCase):
    """! Test the coupled location function (clf::CoupledLocalFunctions) python interface"""
    def test_evaluation(self):
        """! Test evaluating the coupled local function"""
        indim = 5
        outdim = 3
        numPoints = 10
        numLocalPoints = 103
        maxOrder = 4

        para = clf.Parameters()
        para.Add('NumSupportPoints', numPoints)
        para.Add('NumLocalPoints', numLocalPoints)
        para.Add('InputDimension', indim)
        para.Add('OutputDimension', outdim)

        dom = clf.Hypercube(indim)
        multiSet = clf.MultiIndexSet(indim, maxOrder)
        leg = clf.LegendrePolynomials()

        delta = [random.uniform(0.0, 0.1) for i in range(indim)]

        func = clf.CoupledLocalFunctions(multiSet, leg, dom, delta, para)
        self.assertEqual(func.NumLocalFunctions(), numPoints)

        system0 = clf.IdentityModel(indim, outdim)
        system1 = clf.IdentityModel(indim, outdim)
        self.assertNotEqual(system0.id, system1.id)

        numBoundaryPoints = 500
        def Boundary0(samp):
            return samp[1][0]<0 and np.linalg.norm(samp[1][1:])<1.0e-15
        func.AddBoundaryCondition(system1, Boundary0, numBoundaryPoints)
        def Boundary1(samp):
            return samp[1][1]<0 and abs(samp[1][0])<1.0e-15 and np.linalg.norm(samp[1][2:])<1.0e-15
        func.AddBoundaryCondition(system0, Boundary1, numBoundaryPoints)

        for i in range(func.NumLocalFunctions()):
            resids = func.GetResiduals(i);
            if resids is not None:
                for it in resids:
                    self.assertTrue(resids[0].SystemID()<=it.SystemID())

        func.RemoveResidual(system0.id)
        for i in range(func.NumLocalFunctions()):
            resids = func.GetResiduals(i);
            if resids is not None:
                self.assertEqual(len(resids), 1)
                self.assertEqual(resids[0].SystemID(), system1.id)

        advec = clf.AdvectionEquation(indim)
        func.AddResidual(advec, para)
        for i in range(func.NumLocalFunctions()):
            resids = func.GetResiduals(i);
            self.assertTrue(resids is not None)
            for it in resids:
                self.assertTrue(it.SystemID()==system1.id or it.SystemID()==advec.id)
