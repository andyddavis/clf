#ifndef IDENTITYMODEL_HPP_
#define IDENTITYMODEL_HPP_

#include "clf/SystemOfEquations.hpp"

namespace clf {

/// A system of equations with the form \f$u(x) - f(x) = 0\f$, where \f$u(x) = \Phi(x)^{\top} c\f$ 
class IdentityModel : public SystemOfEquations {
public:

  IdentityModel(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~IdentityModel() = default;

  /// Evaluate the operator \f$\mathcal{L}(u(x), x)\f$. given the function \f$u\f$ and at the location \f$x\f$.
  /**
     Defaults to \f$\mathcal{L}(u(x), x) = u(x)\f$.
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const final override;

  /// Evaluate the Jacobian of the operator with respect to the cofficients \f$c\f$, \f$\nabla_{c} \mathcal{L}(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::MatrixXd JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const final override;

  /// Compute the weighted sum of the Hessian of each output of the operator with respect to the cofficients \f$c\f$, \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     @param[in] wieghts The weights for the sum
     \return The operator Hessian \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$
   */
  virtual Eigen::MatrixXd HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const final override;

private:
};

} // namespace clf

#endif
