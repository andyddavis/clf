"""! @brief Test the feature matrix (clf::FeatureMatrix) python interface"""

import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestFeatureMatrix(unittest.TestCase):
    """! Test the feature matrix (clf::FeatureMatrix) python interface"""
    def test_single_feature_vector(self):
        """! Test a feature matrix with only one (repeated) feature vector"""

        indim = int(5)
        outdim = int(3)
        maxOrder = int(4)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('MaximumOrder', maxOrder)

        multiSet = clf.MultiIndexSet(para)
        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        domain = clf.Hypercube(center-np.array([0.1]*indim), center+np.array([0.1]*indim))

        vec = clf.FeatureVector(multiSet, leg)
        mat = clf.FeatureMatrix(vec, outdim, domain)
        self.assertEqual(mat.numBasisFunctions, outdim*multiSet.NumIndices())
        self.assertEqual(mat.InputDimension(), indim)
        self.assertEqual(mat.numFeatureVectors, outdim)

        x = center + np.array([random.uniform(-0.1, 0.1) for i in range(indim)])
        y = domain.MapToHypercube(x)
        fx = vec.Evaluate(y)

        for i in range(outdim):
            self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(i).Evaluate(y)-fx), 0.0)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(mat.numBasisFunctions)])
        output = mat.ApplyTranspose(x, coeff);
        self.assertEqual(len(output), outdim);

        expected = np.zeros(outdim)
        for i in range(outdim):
            expected[i] = np.dot(fx, coeff[i*multiSet.NumIndices():(i+1)*multiSet.NumIndices()])
        self.assertAlmostEqual(np.linalg.norm(output-expected), 0.0)

    def test_multi_feature_vector(self):
        """! Test a feature matrix with multiple (not repeated) feature vectors"""

        indim = int(5)
        outdim = int(2)
        maxOrder1 = int(4)
        maxOrder2 = int(8)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('MaximumOrder', maxOrder1)

        multiSet1 = clf.MultiIndexSet(para)

        para.Add('MaximumOrder', maxOrder2)
        multiSet2 = clf.MultiIndexSet(para)

        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        domain = clf.Hypercube(center-np.array([0.1]*indim), center+np.array([0.1]*indim))

        vec1 = clf.FeatureVector(multiSet1, leg)
        vec2 = clf.FeatureVector(multiSet2, leg)
        mat = clf.FeatureMatrix([vec1, vec2], domain)
        self.assertEqual(mat.numBasisFunctions, multiSet1.NumIndices()+multiSet2.NumIndices())
        self.assertEqual(mat.InputDimension(), indim)
        self.assertEqual(mat.numFeatureVectors, outdim)

        x = center + np.array([random.uniform(-0.1, 0.1) for i in range(indim)])
        y = domain.MapToHypercube(x)
        fx1 = vec1.Evaluate(y)
        fx2 = vec2.Evaluate(y)

        self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(0).Evaluate(y)-fx1), 0.0)
        self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(1).Evaluate(y)-fx2), 0.0)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(mat.numBasisFunctions)])
        output = mat.ApplyTranspose(x, coeff);
        self.assertEqual(len(output), outdim);

        expected = np.zeros(outdim)
        expected[0] = np.dot(fx1, coeff[0:multiSet1.NumIndices()])
        expected[1] = np.dot(fx2, coeff[multiSet1.NumIndices():multiSet1.NumIndices()+multiSet2.NumIndices()])
        self.assertAlmostEqual(np.linalg.norm(output-expected), 0.0)

    def test_repeated_feature_vector(self):
        """! Test a feature matrix with multiple repeated feature vectors"""

        indim = int(5)
        outdim = int(3)
        maxOrder1 = int(4)
        maxOrder2 = int(8)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('MaximumOrder', maxOrder1)

        multiSet1 = clf.MultiIndexSet(para)

        para.Add('MaximumOrder', maxOrder2)
        multiSet2 = clf.MultiIndexSet(para)

        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        domain = clf.Hypercube(center-np.array([0.1]*indim), center+np.array([0.1]*indim))

        vec1 = (clf.FeatureVector(multiSet1, leg), 1)
        vec2 = (clf.FeatureVector(multiSet2, leg), outdim-1)
        mat = clf.FeatureMatrix([vec1, vec2], domain)
        self.assertEqual(mat.numBasisFunctions, multiSet1.NumIndices()+(outdim-1)*multiSet2.NumIndices())
        self.assertEqual(mat.InputDimension(), indim)
        self.assertEqual(mat.numFeatureVectors, outdim)

        x = center + np.array([random.uniform(-0.1, 0.1) for i in range(indim)])
        y = domain.MapToHypercube(x)
        fx1 = vec1[0].Evaluate(y)
        fx2 = vec2[0].Evaluate(y)

        self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(0).Evaluate(y)-fx1), 0.0)
        for i in range(1, outdim):
            self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(i).Evaluate(y)-fx2), 0.0)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(mat.numBasisFunctions)])
        output = mat.ApplyTranspose(x, coeff);
        self.assertEqual(len(output), outdim);

        expected = np.zeros(outdim)
        expected[0] = np.dot(fx1, coeff[0:multiSet1.NumIndices()])
        for i in range(1, outdim):
            expected[i] = np.dot(fx2, coeff[multiSet1.NumIndices()+(i-1)*multiSet2.NumIndices():multiSet1.NumIndices()+i*multiSet2.NumIndices()])
        self.assertAlmostEqual(np.linalg.norm(output-expected), 0.0)
