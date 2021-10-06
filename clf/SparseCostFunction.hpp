#ifndef SPARSECOSTFUNCTION_HPP_
#define SPARSECOSTFUNCTION_HPP_

#include "clf/CostFunction.hpp"

namespace clf {

/// A clf::CostFunction using a sparse Jacobian matrix
class SparseCostFunction : public CostFunction<Eigen::SparseMatrix<double> > {
public:

  /// Create a cost function with \f$m\f$ penalty functions that all have the same output dimension \f$d\f$
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of sub-cost functions \f$m\f$
  @param[in] outputDimension The output dimension \f$d\f$
  */
  SparseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension);

  virtual ~SparseCostFunction() = default;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The entries of the Jacobian of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  std::vector<Eigen::Triplet<double> > PenaltyFunctionJacobianSparse(std::size_t const ind, Eigen::VectorXd const& beta) const;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const final override;

protected:

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The user can no longer override this function. They must override the sparse version.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const final override;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  Default to using finite difference.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  virtual std::vector<Eigen::Triplet<double> > PenaltyFunctionJacobianSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const; 

  /// The sparsity tolerance ignores entries in the Jacobian that are less then this value
  /**
  Defaults to \f$1.0e-14\f$
  */
  const double sparsityTol = 1.0e-14;

private:
};

} // namespace clf

#endif
