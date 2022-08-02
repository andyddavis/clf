"""! @brief Python implemenations of clf::PenaltyFunctions"""

import numpy as np
import scipy

import random

import PyCoupledLocalFunctions as clf

class DensePenaltyFunctionTest0(clf.DensePenaltyFunction):
    """! An example penalty function used to test clf::DensePenaltyFunction"""
    def __init__(self):
        super().__init__(3, 2)

    def Evaluate(self, beta):
        """! Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
        \f{equation*}{
        c(\beta) = \begin{bmatrix}
        \beta_0 \\
        \beta_0 (1-\beta_2)
        \end{bmatrix}.
    \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
        @param[in] beta The input parameters \f$\beta\f$
        \return The penalty function evaluation \f$c(\beta)\f$
        """
        return [beta[0], beta[0]*(1.0-beta[2])]

    def Jacobian(self, beta):
        """! Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
        \f{equation*}{
        \nabla_{\beta} c(\beta) = \begin{bmatrix}
        1 & 0 & 0 \\
        1-\beta_2 & 0 & -\beta_0 
        \end{bmatrix} \in \mathbb{R}^{2 \times 3}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return The gradient of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$
        """
        jac = np.zeros((2, 3))
        jac[0, 0] = 1.0
        jac[1, 0] = 1.0-beta[2]
        jac[1, 2] = -beta[0]
        return jac

    def Hessian(self, beta, weights):
        """! Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the first component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}
        \f}
        and the Hessian of the second component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
        0 & 0 & -1 \\ 
        0 & 0 & 0 \\ 
        -1 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        This returns the sum of these Hessians.
        @param[in] beta The input parameters \f$\beta\f$
        @param[in] weights The weights for the weighted sum
        \return The sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
        """
        hess = np.zeros((3, 3))
        hess[0, 2] = -weights[1]
        hess[2, 0] = -weights[1]
        return hess

class DensePenaltyFunctionTest1(clf.DensePenaltyFunction):
    """! An example penalty function used to test clf::DensePenaltyFunction"""
    def __init__(self):
        super().__init__(3, 6)

    def Evaluate(self, beta):
        """! Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
    
        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
        \f{equation*}{
        c(\beta) = \begin{bmatrix}
        1-\beta_1 \\
        1-\beta_1+\beta_2 \\
        \beta_2 \\
        \beta_2(1-\beta_1) \\
        \beta_0 \beta_2 \\
        \beta_0^2 \beta_1
        \end{bmatrix}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
        @param[in] beta The input parameters \f$\beta\f$
        \return The penalty function evaluation \f$c(\beta)\f$
        """
        return [1.0-beta[1],
                1.0-beta[1]+beta[2],
                beta[2],
                beta[2]*(1.0-beta[1]),
                beta[0]*beta[2],
                beta[0]*beta[0]*beta[1]]

    def Jacobian(self, beta):
        """! Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{6 \times 3}\f$
        
        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
        \f{equation*}{
        \nabla_{\beta} c(\beta) = \begin{bmatrix}
        0 & -1 & 0 \\
        0 & -1 & 1 \\
        0 & 0 & 1 \\
        0 & -\beta_2 & 1-\beta_1 \\
        \beta_2 & 0 & \beta_0 \\
        2 \beta_0 \beta_1 & \beta_0^2 & 0
        \end{bmatrix} \in \mathbb{R}^{6 \times 3}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$
        """
        return [[0.0, -1.0, 0.0],
                [0.0, -1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, -beta[2], 1.0-beta[1]],
                [beta[2], 0.0, beta[0]],
                [2.0*beta[0]*beta[1], beta[0]*beta[0], 0.0]]

    def Hessian(self, beta, weights):
        """! Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
        
        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the \f$1^{\text{st}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}
        \f}
        and the Hessian of the \f$2^{\text{nd}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\ 
        0 & 0 & 0 \\ 
        0 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$3^{\text{rd}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_3(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\ 
        0 & 0 & 0 \\ 
        0 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$4^{\text{th}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_4(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\ 
        0 & 0 & -1 \\ 
        0 & -1 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$5^{\text{th}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_5(\beta) = \begin{bmatrix}
        0 & 0 & 1 \\ 
        0 & 0 & 0 \\ 
        1 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$6^{\text{th}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_6(\beta) = \begin{bmatrix}
        2 \beta_1 & 2 \beta_0 & 0 \\ 
        2 \beta_0 & 0 & 0 \\ 
        0 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        This returns the sum of these Hessians.
        @param[in] beta The input parameters \f$\beta\f$
        \return The sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
        """
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
    """! An example penalty function used to test clf::SparsePenaltyFunction"""
    def __init__(self):
        super().__init__(3, 2)

    def Evaluate(self, beta):
        """! Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
        
        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
        \f{equation*}{
        c(\beta) = \begin{bmatrix}
        \beta_0 \\
        \beta_0 (1-\beta_2)
        \end{bmatrix}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
        @param[in] beta The input parameters \f$\beta\f$
        \return The penalty function evaluation \f$c(\beta)\f$
        """
        return [beta[0], beta[0]*(1.0-beta[2])]

    def JacobianEntries(self, beta):
        """! Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
        \f{equation*}{
        \nabla_{\beta} c(\beta) = \begin{bmatrix}
        1 & 0 & 0 \\
        1-\beta_2 & 0 & -\beta_0 
        \end{bmatrix} \in \mathbb{R}^{2 \times 3}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return The Jacobian matrix entries
        """
        return [clf.SparseEntry(0, 0, 1.0), clf.SparseEntry(1, 0, 1.0-beta[2]), clf.SparseEntry(1, 2, -beta[0])]

    def HessianEntries(self, beta, weights):
        """! Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the first component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}
        \f}
        and the Hessian of the second component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
        0 & 0 & -1 \\ 
        0 & 0 & 0 \\ 
        -1 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        This returns the sum of these Hessians.
        @param[in] beta The input parameters \f$\beta\f$
        @param[in] weights The weights for the weighted sum
        \return The entries Hessian matrix
        """
        return [clf.SparseEntry(0, 2, -weights[1]), clf.SparseEntry(2, 0, -weights[1])]

class SparsePenaltyFunctionTest1(clf.SparsePenaltyFunction):
    """! An example penalty function used to test clf::SparsePenaltyFunction"""
    def __init__(self):
        super().__init__(3, 6)

    def Evaluate(self, beta):
        """! Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
        \f{equation*}{
        c(\beta) = \begin{bmatrix}
        1-\beta_1 \\
        1-\beta_1+\beta_2 \\
        \beta_2 \\
        \beta_2(1-\beta_1) \\
        \beta_0 \beta_2 \\
        \beta_0^2 \beta_1
        \end{bmatrix}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
        @param[in] beta The input parameters \f$\beta\f$
        \return The penalty function evaluation \f$c(\beta)\f$
        """
        return [1.0-beta[1],
                1.0-beta[1]+beta[2],
                beta[2],
                beta[2]*(1.0-beta[1]),
                beta[0]*beta[2],
                beta[0]*beta[0]*beta[1]]

    def JacobianEntries(self, beta):
        """! Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{6 \times 3}\f$

        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
        \f{equation*}{
        \nabla_{\beta} c(\beta) = \begin{bmatrix}
        0 & -1 & 0 \\
        0 & -1 & 1 \\
        0 & 0 & 1 \\
        0 & -\beta_2 & 1-\beta_1 \\
        \beta_2 & 0 & \beta_0 \\
        2 \beta_0 \beta_1 & \beta_0^2 & 0
        \end{bmatrix} \in \mathbb{R}^{6 \times 3}.
        \f}
        @param[in] beta The input parameters \f$\beta\f$
        \return The entries of Jacobian matrix
        """
        return [clf.SparseEntry(0, 1, -1.0),
                clf.SparseEntry(1, 1, -1.0),
                clf.SparseEntry(1, 2, 1.0),
                clf.SparseEntry(2, 2, 1.0),
                clf.SparseEntry(3, 1, -beta[2]),
                clf.SparseEntry(3, 2, 1.0-beta[1]),
                clf.SparseEntry(4, 0, beta[2]),
                clf.SparseEntry(4, 2, beta[0]),
                clf.SparseEntry(5, 0, 2.0*beta[0]*beta[1]),
                clf.SparseEntry(5, 1, beta[0]*beta[0])]

    def HessianEntries(self, beta, weights):
        """! Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
        
        The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the \f$1^{\text{st}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}
        \f}
        and the Hessian of the \f$2^{\text{nd}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\ 
        0 & 0 & 0 \\ 
        0 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$3^{\text{rd}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_3(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\ 
        0 & 0 & 0 \\ 
        0 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$4^{\text{th}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_4(\beta) = \begin{bmatrix}
        0 & 0 & 0 \\ 
        0 & 0 & -1 \\ 
        0 & -1 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$5^{\text{th}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_5(\beta) = \begin{bmatrix}
        0 & 0 & 1 \\ 
        0 & 0 & 0 \\ 
        1 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        and the Hessian of the \f$6^{\text{th}}\f$ component of the the penalty function is
        \f{equation*}{
        \nabla_{\beta}^2 c_6(\beta) = \begin{bmatrix}
        2 \beta_1 & 2 \beta_0 & 0 \\ 
        2 \beta_0 & 0 & 0 \\ 
        0 & 0 & 0 
        \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
        \f}
        This returns the sum of these Hessians.
        @param[in] beta The input parameters \f$\beta\f$
        @param[in] weights The weights for the weighted sum
        \return The Hessian matrix
        """
        return [clf.SparseEntry(1, 2, -weights[3]),
                clf.SparseEntry(2, 1, -weights[3]),
                clf.SparseEntry(0, 2, weights[4]),
                clf.SparseEntry(2, 0, weights[4]),
                clf.SparseEntry(0, 0, 2.0*beta[1]*weights[5]),
                clf.SparseEntry(0, 1, 2.0*beta[0]*weights[5]),
                clf.SparseEntry(1, 0, 2.0*beta[0]*weights[5])]
