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

  /// Compute the Hessian of the cost function
  /**
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} \left( \sum_{j=1}^{d_i} f_i^{(j)}(\beta) \nabla_{\beta}^2 f_i^{(j)}(\beta) + (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta) \right), 
  \f}
  where \f$\nabla_{\beta}^2 f_i^{(j)}(\beta)\f$ is the Hessian of the \f$j^{th}\f$ output of \f$f_i\f$. Alternatively, we could compute the Gauss-Newton approximation
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  @param[in] beta The current parameter value
  @param[in] gn <tt>true</tt>: Compute the Gauss-Newton Hessian, <tt>false</tt> (default): Compute the full Hessian 
  \return The Hessian matrix (or Gauss-Newton Hessian)
  */
  virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& beta, bool const gn) override;

private:
};

} // namespace clf

#endif
