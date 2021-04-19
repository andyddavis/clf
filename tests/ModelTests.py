import unittest

import numpy as np

import CoupledLocalFunctions as clf

class TestModelForPythonComponentInterface(clf.Model):
    def __init__(self, options):
        clf.Model.__init__(self, options)

    def RightHandSideComponentImpl(self, x, outind):
        return x[0]+outind

class TestModelForPythonVectorInterface(clf.Model):
    def __init__(self, options):
        clf.Model.__init__(self, options)

    def RightHandSideVectorImpl(self, x):
        return [x[0]]*self.outputDimension

class TestModel(unittest.TestCase):
    def test_right_hand_side_evaluation_component_impl(self):
        indim = 2
        outdim = 3

        options = dict()
        options["InputDimension"] = indim
        options["OutputDimension"] = outdim
        model = TestModelForPythonComponentInterface(options)
        x = np.random.rand(indim)
        rhs = model.RightHandSide(x)
        self.assertEqual(len(rhs), outdim)
        for i in range(outdim):
            self.assertAlmostEqual(rhs[i], x[0]+i, 1.0e-12)


    def test_right_hand_side_evaluation_vector_impl(self):
        indim = 2
        outdim = 3

        options = dict()
        options["InputDimension"] = indim
        options["OutputDimension"] = outdim
        model = TestModelForPythonVectorInterface(options)
        x = np.random.rand(indim)
        rhs = model.RightHandSide(x)
        self.assertEqual(len(rhs), outdim)
        for i in range(outdim):
            self.assertAlmostEqual(rhs[i], x[0], 1.0e-12)
