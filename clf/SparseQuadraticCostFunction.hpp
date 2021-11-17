#ifndef SPARSEQUADRATICCOSTFUNCTION_HPP_
#define SPARSEQUADRATICCOSTFUNCTION_HPP_

#include "clf/QuadraticCostFunction.hpp"

namespace clf {

/// A clf::QuadraticCostFunction with a sparse Jacobian matrix
class SparseQuadraticCostFunction : public QuadraticCostFunction<Eigen::SparseMatrix<double> > {
public:

  /// Create a cost function with \f$m\f$ penalty functions that all have the same output dimension \f$d\f$
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  @param[in] outputDimension The output dimension \f$d\f$
  */
  SparseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension);

  virtual ~SparseQuadraticCostFunction() = default;

  /// Compute the Jacobian matrix \f$A_i\f$ of the linear penalty function \f$f_i\f$
  /**
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  std::vector<Eigen::Triplet<double> > PenaltyFunctionJacobianSparse(std::size_t const ind) const;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::SparseMatrix<double>& jac) const final override;

protected:

  /// Compute the Jacobian matrix \f$A_i\f$ of the linear penalty function \f$f_i\f$
  /**
  The user can no longer override this function. They must override the sparse version.
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind) const final override;

  /// Compute the Jacobian matrix \f$A_i\f$ of the linear penalty function \f$f_i\f$
  /**
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  virtual std::vector<Eigen::Triplet<double> > PenaltyFunctionJacobianSparseImpl(std::size_t const ind) const = 0;

  /// Apply the Jacobian matrix of the \f$i^{th}\f$ penalty function to the parameters 
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The application of the \f$i^{th}\f$ penalty function's Jacobian matrix on \f$\beta\f$
  */
  virtual Eigen::VectorXd ApplyPenaltyFunctionJacobian(std::size_t const ind, Eigen::VectorXd const& beta) const override;

  /// Evaluate the Hessian \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$ of the penalty function
  /**
  If the cost function is quadratic, then the penalty functions must be linear. Therefore, the penalty function Hessian is zero.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter \f$\beta \in \mathbb{R}^{n}\f$
  @param[in] order The order of the finite difference approximation
  @param[in] dbeta The \f$\Delta \beta\f$ used to compute finite difference approximations (defaults to \f$1e-8\f$)
  \return Each component is the Hessian of the \f$j^{th}\f$ couput of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$
  */
  virtual std::vector<Eigen::SparseMatrix<double> > PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order = FIRST_UPWARD, double const dbeta = 1.0e-8) const final override;

protected:

  /// Evaluate the Hessian \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$ of the penalty function
  /**
  Since the cost function is quadratic, the penalty function must be linear. Therefore, its Hessian is zero.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter \f$\beta \in \mathbb{R}^{n}\f$
  \return Each component is the Hessian of the \f$j^{th}\f$ couput of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$
  */
  virtual std::vector<Eigen::SparseMatrix<double> > PenaltyFunctionHessianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const final override;

private:
};

} // namespace clf

#endif
