#ifndef DENSECOSTFUNCTION_HPP_
#define DENSECOSTFUNCTION_HPP_

#include "clf/CostFunction.hpp"

namespace clf {

/// A clf::CostFunction using a dense Jacobian matrix 
class DenseCostFunction : public CostFunction<Eigen::MatrixXd> {
public:

  /// Create a cost function with \f$m\f$ penalty functions that all have the same output dimension \f$d\f$
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of sub-cost functions \f$m\f$
  @param[in] outputDimension The output dimension \f$d\f$
  */
  DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension);

  virtual ~DenseCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const final override;

private:
};

} // namespace clf

#endif
