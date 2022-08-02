"""! @brief Test the feature vector (clf::FeatureVector) python interface"""

import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestFeatureVector(unittest.TestCase):
    """! Test the feature vector (clf::FeatureVector) python interface"""
    def test_evaluate(self):
        """! Test the feature vector evaluation"""
        
        indim = int(5)
        outdim = int(3)
        maxOrder = int(4)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('MaximumOrder', maxOrder)
        para.Add('LocalRadius', 1.0)

        multiSet = clf.MultiIndexSet(para)
        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])

        vec = clf.FeatureVector(multiSet, leg, center, para)
        self.assertEqual(vec.InputDimension(), multiSet.Dimension())
        self.assertEqual(vec.NumBasisFunctions(), multiSet.NumIndices())

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        y = vec.Transformation(x)
        fx = vec.Evaluate(x)

        expected = np.array([1.0]*multiSet.NumIndices())
        for i in range(multiSet.NumIndices()): 
            for d in range(indim): 
                expected[i] *= leg.Evaluate(multiSet.indices[i].alpha[d], y[d])

        self.assertEqual(len(fx), len(expected))
        self.assertAlmostEqual(np.linalg.norm(fx-expected), 0.0)
