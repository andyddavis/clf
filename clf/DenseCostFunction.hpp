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

  /// Create a cost function with \f$m\f$ penalty terms that all have different output dimension
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  @param[in] outputDimensions Each component indicates there is <tt>outputDimension[i].first</tt> penalty functions with dimension <tt>outputDimension[i].second</tt>
  */
  DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::vector<std::pair<std::size_t, std::size_t> > const& outputDimension);

  virtual ~DenseCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const final override;

  /// Evaluate the Hessian \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$ of the penalty function
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter \f$\beta \in \mathbb{R}^{n}\f$
  @param[in] order The order of the finite difference approximation
  @param[in] dbeta The \f$\Delta \beta\f$ used to compute finite difference approximations (defaults to \f$1e-8\f$)
  \return Each component is the Hessian of the \f$j^{th}\f$ couput of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$
  */
  virtual std::vector<Eigen::MatrixXd> PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order = FIRST_UPWARD, double const dbeta = 1.0e-8) const final override;

private:
};

} // namespace clf

#endif
