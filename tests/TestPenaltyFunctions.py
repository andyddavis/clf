import numpy as np
import scipy

import random

import PyCoupledLocalFunctions as clf

class DensePenaltyFunctionTest0(clf.DensePenaltyFunction):
    def __init__(self):
        super().__init__(3, 2)

    def Evaluate(self, beta):
        return [beta[0], beta[0]*(1.0-beta[2])]

    def Jacobian(self, beta):
        jac = np.zeros((2, 3))
        jac[0, 0] = 1.0
        jac[1, 0] = 1.0-beta[2]
        jac[1, 2] = -beta[0]
        return jac

    def Hessian(self, beta, weights):
        hess = np.zeros((3, 3))
        hess[0, 2] = -weights[1]
        hess[2, 0] = -weights[1]
        return hess

class DensePenaltyFunctionTest1(clf.DensePenaltyFunction):
    def __init__(self):
        super().__init__(3, 6)

    def Evaluate(self, beta):
        return [1.0-beta[1],
                1.0-beta[1]+beta[2],
                beta[2],
                beta[2]*(1.0-beta[1]),
                beta[0]*beta[2],
                beta[0]*beta[0]*beta[1]]

    def Jacobian(self, beta):
        return [[0.0, -1.0, 0.0],
                [0.0, -1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, -beta[2], 1.0-beta[1]],
                [beta[2], 0.0, beta[0]],
                [2.0*beta[0]*beta[1], beta[0]*beta[0], 0.0]]

    def Hessian(self, beta, weights):
        hess = np.zeros((3, 3))
        hess[1, 2] = -weights[3]
        hess[2, 1] = -weights[3]
        hess[0, 2] = weights[4]
        hess[2, 0] = weights[4]
        hess[0, 0] = 2.0*beta[1]*weights[5]
        hess[0, 1] = 2.0*beta[0]*weights[5]
        hess[1, 0] = 2.0*beta[0]*weights[5]
        return hess

class SparsePenaltyFunctionTest0(clf.SparsePenaltyFunction):
    def __init__(self):
        super().__init__(3, 2)

    def Evaluate(self, beta):
        return [beta[0], beta[0]*(1.0-beta[2])]

    def Jacobian(self, beta):
        row = [0, 1, 1]
        col = [0, 0, 2]
        data = [1.0, 1.0-beta[2], -beta[0]]
        return scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 3))

    def Hessian(self, beta, weights):
        row = [0, 2]
        col = [2, 0]
        data = [-weights[1], -weights[1]]
        return scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))

class SparsePenaltyFunctionTest1(clf.SparsePenaltyFunction):
    def __init__(self):
        super().__init__(3, 6)

    def Evaluate(self, beta):
        return [1.0-beta[1],
                1.0-beta[1]+beta[2],
                beta[2],
                beta[2]*(1.0-beta[1]),
                beta[0]*beta[2],
                beta[0]*beta[0]*beta[1]]

    def Jacobian(self, beta):
        row = [0, 1, 1, 2, 3, 3, 4, 4, 5, 5]
        col = [1, 1, 2, 2, 1, 2, 0, 2, 0, 1]
        data = [-1.0, -1.0, 1.0, 1.0, -beta[2], 1.0-beta[1], beta[2], beta[0], 2.0*beta[0]*beta[1], beta[0]*beta[0]]
        return scipy.sparse.csr_matrix((data, (row, col)), shape=(6, 3))

    def Hessian(self, beta, weights):
        row = [1, 2, 0, 2, 0, 0, 1]
        col = [2, 1, 2, 0, 0, 1, 0]
        data = [-weights[3], -weights[3], weights[4], weights[4], 2.0*beta[1]*weights[5], 2.0*beta[0]*weights[5], 2.0*beta[0]*weights[5]]
        return scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))
