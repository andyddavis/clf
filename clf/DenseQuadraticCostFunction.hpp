#ifndef DENSEQUADRATICCOSTFUNCTION_HPP_
#define DENSEQUADRATICCOSTFUNCTION_HPP_

#include "clf/QuadraticCostFunction.hpp"

namespace clf {

/// A clf::QuadraticCostFunction with a dense Jacobian matrix
class DenseQuadraticCostFunction : public QuadraticCostFunction<Eigen::MatrixXd> {
public:

  /// Create a cost function with \f$m\f$ penalty functions that all have the same output dimension \f$d\f$
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  @param[in] outputDimension The output dimension \f$d\f$
  */
  DenseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension);

  virtual ~DenseQuadraticCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the penalty function \f$f_i\f$ with respect to the input parameters \f$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionGradientImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::MatrixXd& jac) const final override;

protected:

private:
};

} // namespace clf 

#endif
