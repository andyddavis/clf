import unittest

import numpy as np

import CoupledLocalFunctions as clf

class TestSupportPointCloud(unittest.TestCase):
    def test_create(self):
        # options for the basis
        basisOptions = dict()
        basisOptions["Type"] = "TotalOrderPolynomials"
        basisOptions["Order"] = 3

        # options for the model
        modelOptions = dict()
        modelOptions['InputDimension'] = 2
        modelOptions['OutputDimension'] = 1
        model = clf.Model(modelOptions)

        # options for the support point
        ptOptions = dict()
        ptOptions["BasisFunctions"] = "Basis"
        ptOptions["Basis"] = basisOptions

        # the number of support points
        npoints = 50
        points = [None]*npoints
        for i in range(npoints):
            # create the point
            points[i] = clf.SupportPoint(np.random.rand(2), model, ptOptions)

        # make the point cloud
        cloudOptions = dict()
        cloud = clf.SupportPointCloud(points, cloudOptions)

        self.assertEqual(cloud.NumSupportPoints(), npoints)
        self.assertEqual(cloud.InputDimension(), 2)
        self.assertEqual(cloud.OutputDimension(), 1)
        for i in range(npoints):
            self.assertAlmostEqual(np.linalg.norm(cloud.GetSupportPoint(i).x-points[i].x), 0.0, places=12)

        neighInd, neighDist = cloud.FindNearestNeighbors(np.random.rand(2), 5)
        self.assertEqual(len(neighInd), 5)
        self.assertEqual(len(neighDist), 5)
        for ind in neighInd:
            self.assertTrue(ind<npoints)
