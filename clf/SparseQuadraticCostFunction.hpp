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

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparse(std::size_t const ind) const;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::SparseMatrix<double>& jac) const final override;

protected:

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The user can no longer override this function. They must override the sparse version.
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind) const final override;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  Default to using finite difference with \f$\beta=0\f$. By children can implement more efficient gradient computations
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  virtual std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparseImpl(std::size_t const ind) const;

  /// The sparsity tolerance ignores entries in the Jacobian that are less then this value 
  /**
  Defaults to \f$1.0e-14\f$
  */
  const double sparsityTol = 1.0e-14;

private:
};

} // namespace clf 

#endif
