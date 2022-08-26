#ifndef DENSEPENALTYFUNCTION_HPP_
#define DENSEPENALTYFUNCTION_HPP_

#include "clf/PenaltyFunction.hpp"

namespace clf {

/// A penalty function (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
class DensePenaltyFunction : public PenaltyFunction<Eigen::MatrixXd> {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
     @param[in] para The parameters for this penalty function 
  */
  DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());
  
  virtual ~DensePenaltyFunction() = default;
  
  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$
   */
  virtual Eigen::MatrixXd JacobianFD(Eigen::VectorXd const& beta) final override;

  /// Compute the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  virtual Eigen::MatrixXd HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) final override;
  
private:
};

} // namespace clf

#endif
