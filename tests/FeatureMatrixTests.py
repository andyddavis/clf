import unittest

import numpy as np
import random

import PyCoupledLocalFunctions as clf

class TestFeatureMatrix(unittest.TestCase):
    def test_single_feature_vector(self):
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
        mat = clf.FeatureMatrix(vec, outdim)
        self.assertEqual(mat.numBasisFunctions, outdim*multiSet.NumIndices())
        self.assertEqual(mat.InputDimension(), indim)
        self.assertEqual(mat.numFeatureVectors, outdim)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        fx = vec.Evaluate(x)

        for i in range(outdim):
            self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(i).Evaluate(x)-fx), 0.0)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(mat.numBasisFunctions)])
        output = mat.ApplyTranspose(x, coeff);
        self.assertEqual(len(output), outdim);

        expected = np.zeros(outdim)
        for i in range(outdim):
            expected[i] = np.dot(fx, coeff[i*multiSet.NumIndices():(i+1)*multiSet.NumIndices()])
        self.assertAlmostEqual(np.linalg.norm(output-expected), 0.0)

    def test_multi_feature_vector(self):
        indim = int(5)
        outdim = int(2)
        maxOrder1 = int(4)
        maxOrder2 = int(8)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('MaximumOrder', maxOrder1)
        para.Add('LocalRadius', 1.0)

        multiSet1 = clf.MultiIndexSet(para)

        para.Add('MaximumOrder', maxOrder2)
        multiSet2 = clf.MultiIndexSet(para)

        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])

        vec1 = clf.FeatureVector(multiSet1, leg, center, para)
        vec2 = clf.FeatureVector(multiSet2, leg, center, para)
        mat = clf.FeatureMatrix([vec1, vec2])
        self.assertEqual(mat.numBasisFunctions, multiSet1.NumIndices()+multiSet2.NumIndices())
        self.assertEqual(mat.InputDimension(), indim)
        self.assertEqual(mat.numFeatureVectors, outdim)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        fx1 = vec1.Evaluate(x)
        fx2 = vec2.Evaluate(x)

        self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(0).Evaluate(x)-fx1), 0.0)
        self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(1).Evaluate(x)-fx2), 0.0)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(mat.numBasisFunctions)])
        output = mat.ApplyTranspose(x, coeff);
        self.assertEqual(len(output), outdim);

        expected = np.zeros(outdim)
        expected[0] = np.dot(fx1, coeff[0:multiSet1.NumIndices()])
        expected[1] = np.dot(fx2, coeff[multiSet1.NumIndices():multiSet1.NumIndices()+multiSet2.NumIndices()])
        self.assertAlmostEqual(np.linalg.norm(output-expected), 0.0)

    def test_repeated_feature_vector(self):
        indim = int(5)
        outdim = int(3)
        maxOrder1 = int(4)
        maxOrder2 = int(8)
    
        para = clf.Parameters()
        para.Add('InputDimension', indim)
        para.Add('MaximumOrder', maxOrder1)
        para.Add('LocalRadius', 1.0)

        multiSet1 = clf.MultiIndexSet(para)

        para.Add('MaximumOrder', maxOrder2)
        multiSet2 = clf.MultiIndexSet(para)

        leg = clf.LegendrePolynomials()
        center = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])

        vec1 = (clf.FeatureVector(multiSet1, leg, center, para), 1)
        vec2 = (clf.FeatureVector(multiSet2, leg, center, para), outdim-1)
        mat = clf.FeatureMatrix([vec1, vec2])
        self.assertEqual(mat.numBasisFunctions, multiSet1.NumIndices()+(outdim-1)*multiSet2.NumIndices())
        self.assertEqual(mat.InputDimension(), indim)
        self.assertEqual(mat.numFeatureVectors, outdim)

        x = np.array([random.uniform(-1.0, 1.0) for i in range(indim)])
        fx1 = vec1[0].Evaluate(x)
        fx2 = vec2[0].Evaluate(x)

        self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(0).Evaluate(x)-fx1), 0.0)
        for i in range(1, outdim):
            self.assertAlmostEqual(np.linalg.norm(mat.GetFeatureVector(i).Evaluate(x)-fx2), 0.0)

        coeff = np.array([random.uniform(-1.0, 1.0) for i in range(mat.numBasisFunctions)])
        output = mat.ApplyTranspose(x, coeff);
        self.assertEqual(len(output), outdim);

        expected = np.zeros(outdim)
        expected[0] = np.dot(fx1, coeff[0:multiSet1.NumIndices()])
        for i in range(1, outdim):
            expected[i] = np.dot(fx2, coeff[multiSet1.NumIndices()+(i-1)*multiSet2.NumIndices():multiSet1.NumIndices()+i*multiSet2.NumIndices()])
        self.assertAlmostEqual(np.linalg.norm(output-expected), 0.0)
