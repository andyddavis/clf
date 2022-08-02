"""! @brief Test the local residual (clf::LocalResidual) python interface"""

import unittest
import random
import numpy as np

import PyCoupledLocalFunctions as clf

class TestLocalResidual(unittest.TestCase):
    """! @brief Test the local residual (clf::LocalResidual) python interface"""

    def setUp(self):
        """! Set up the input and output dimension"""
        ## The input dimension
        self.indim = 3
        ## The output dimension
        self.outdim = 2

    def tearDown(self):
        """! Test the local residual python interface"""

        self.assertTrue(False)

    def test_identity_model(self):
        """! Test using and identity model"""
        ## The matrix that defines the model
        self.mat = np.identity(self.outdim);
        ## The model
        self.system = clf.IdentityModel(self.indim, self.outdim)

    def test_linear_model(self):
        """! Test using and linear model"""
        ## The matrix that defines the model
        self.mat = np.zeros((self.outdim, self.outdim));
        for i in range(self.outdim):
            for j in range(self.outdim):
                self.mat[i, j] = random.uniform(-1.0, 1.0)

        ## The model
        self.system = clf.LinearModel(self.indim, self.mat)
