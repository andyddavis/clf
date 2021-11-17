#ifndef QUADRATICCOSTFUNCTION_HPP_
#define QUADRATICCOSTFUNCTION_HPP_

#include "clf/CostFunction.hpp"

namespace clf {

/// A clf::CostFunction with linear penalty functions
/**
The cost function is the sum of \f$m\f$ squared penalty functions \f$f_i: \mathbb{R}^{n} \mapsto \mathbb{R}^{d_i}\f$,
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} \| f_i(\boldsymbol{\beta}) \|^{2}.
\f}
In the case of a quadratic cost function the penalty functions \f$f_i\f$ are all linear with respect to \f$\beta\f$, implying that \f$f_i(\beta) = A_i \beta - g_i\f$ for \f$A \in \mathbb{R}^{d_i \times n}\f$ and \f$g_i \in \mathbb{R}^{d_i}\f$. The matrix \f$A_i\f$ as the Jacobian of \f$f_i\f$ and \f$g_i\f$ is the right hand side.
*/
template<typename MatrixType>
class QuadraticCostFunction : public CostFunction<MatrixType> {
public:

  /// Create a cost function with \f$m\f$ penalty functions that all have the same output dimension \f$d\f$
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  @param[in] outputDimension The output dimension \f$d\f$
  */
  inline QuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : CostFunction<MatrixType>(inputDimension, numPenaltyFunctions, outputDimension) {}

  virtual ~QuadraticCostFunction() = default;

  /// Compute the Jacobian matrix \f$A_i\f$ of the linear penalty function \f$f_i\f$
  /**
  The gradient should be independent of \f$\beta\f$ since \f$f_i\f$ is linear in this case.
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$ (equivalently, \f$A_i\f$)
  */
  inline Eigen::MatrixXd PenaltyFunctionJacobian(std::size_t const ind) const {
    assert(ind<this->numPenaltyFunctions);
    const Eigen::MatrixXd jac = PenaltyFunctionJacobianImpl(ind);
    assert(jac.cols()==this->inputDimension);
    return jac;
  }

  /// Return the right hand side \f$g_i\f$ of the linear penalty function \f$f_i\f$
  /**
  By default, return the zero vector for the right hand side function.
  @param[in] ind The index of the penalty function
  \return The right hand side \f$g_i\f$
  */
  inline Eigen::VectorXd PenaltyFunctionRHS(std::size_t const ind) const {
    assert(ind<this->numPenaltyFunctions);
    return PenaltyFunctionRHSImpl(ind);
  }

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the penalty function \f$f_i\f$ with respect to the input parameters \f$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionJacobianImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(MatrixType& jac) const = 0;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const final override { Jacobian(jac); }

  /// Compute the Hessian of the cost function
  /**
  The Hessian of the cost function is
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  since the penalty functions are linear. 
  @param[in] beta The current parameter value
  @param[out] hess The Hessian matrix
  @param[in] gn <tt>true</tt>: Compute the Gauss-Newton Hessian, <tt>false</tt> (default): Compute the full Hessian; In this case the Gauss-Newton Hessian is exact so this flag acutally does nothing.
  */
  inline virtual void Hessian(Eigen::VectorXd const& beta, MatrixType& hess, bool const gn = false) final override { this->GaussNewtonHessian(beta, hess); }
  
  /// Is this a quadratic cost function? Yes, it is.
  /**
  \return <tt>true</tt>: The cost function is quadratic
  */
  virtual inline bool IsQuadratic() const final override { return true; }

protected:

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i: \mathbb{R}^{n} \mapsto \mathbb{R}^{d_i}\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  inline virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const final override { return ApplyPenaltyFunctionJacobian(ind, beta) - PenaltyFunctionRHS(ind); }

  /// Apply the Jacobian matrix of the \f$i^{th}\f$ penalty function to the parameters 
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The application of the \f$i^{th}\f$ penalty function's Jacobian matrix on \f$\beta\f$
  */
  inline virtual Eigen::VectorXd ApplyPenaltyFunctionJacobian(std::size_t const ind, Eigen::VectorXd const& beta) const { return PenaltyFunctionJacobian(ind)*beta; }

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The gradient should be independent of \f$\beta\f$ since \f$f_i\f$ is linear in this case. Therefore, override this function to just call the implementation without \f$\beta\f$.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const final override { return PenaltyFunctionJacobianImpl(ind); }

  /// Return the right hand side \f$g_i\f$ of the linear penalty function \f$f_i\f$
  /**
  By default, return the zero vector for the right hand side function.
  @param[in] ind The index of the penalty function
  \return The right hand side \f$g_i\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionRHSImpl(std::size_t const ind) const {
    assert(ind<this->numPenaltyFunctions);
    const std::size_t outputDimension = this->PenaltyFunctionOutputDimension(ind);
    return Eigen::VectorXd::Zero(outputDimension);
  }

  /// Evaluate the Jacobian \f$A_i\f$ of the \f$i^{th}\f$ penalty function
  /**
  This function computes the gradient using finite difference around \f$\beta=0\f$. More efficient gradient calculation can be implemented by children.
  @param[in] ind The index of the penalty function
  \return The Jacobian of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind) const = 0;

private:
};

} // namespace clf

#endif
