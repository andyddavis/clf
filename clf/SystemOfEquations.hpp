#ifndef SYSTEMOFEQUATIONS_HPP_
#define SYSTEMOFEQUATIONS_HPP_

#include "clf/Parameters.hpp"

#include "clf/LocalFunction.hpp"

namespace clf {

/// A system of equations with the form \f$\mathcal{L}(u(\cdot), x) - f(x) = 0\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$ and \f$u(x) = \Phi(x)^{\top} c\f$.
/**
   Let \f$u: \Omega \mapsto \mathbb{R}^{m}\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$. The system of equations is composed of two parts. First, \f$\mathcal{L}(u(x), x)\f$ is the operator and, second, \f$f(x)\f$ is the right hand side.
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "DeltaFD"   | <tt>double</tt> | <tt>1.0e-2</tt> | The step size for the finite difference approximation (see SystemOfEquations::deltaFD_DEFAULT). |
   "OrderFD"   | <tt>std::size_t</tt> | <tt>8</tt> | The accuracy order for the finite difference approximation (see SystemOfEquations::orderFD_DEFAULT). The options are \f$2\f$, \f$4\f$, \f$6\f$, and \f$8\f$. |
 */
class SystemOfEquations {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$m\f$
     @param[in] para The parameters for this system of equations
   */
  SystemOfEquations(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension \f$d\f$ of the local function. This is a required parameter. |
   "OutputDimension"   | <tt>std::size_t</tt> | --- | The output dimension \f$m\f$ of the local function. This is a required parameter. |
     @param[in] para The parameters for this system of equations
   */
  SystemOfEquations(std::shared_ptr<const Parameters> const& para);
  
  virtual ~SystemOfEquations() = default;

  /// Evaluate the right hand side at the location \f$x\f$.
  /**
     Defaults to \f$f(x)=0\f$.
     @param[in] x The location \f$x\f$
     \return The right hand side \f$f(x)\f$
   */
  virtual Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const;

  /// Evaluate the operator \f$\mathcal{L}(u(x), x)\f$. given the function \f$u\f$ and at the location \f$x\f$.
  /**
     This must be implemented, the default implementation throws and exception.
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the local function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const;

  /// Compute the Jacobian of the operator with respect to the cofficients \f$c\f$, \f$\nabla_{c} \mathcal{L}(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     Defaults to approximating the Jacobian with finite difference (see SystemOfEquations::JacobianWRTCoefficientsFD)
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator Jacobian \f$\nabla_{c} \mathcal{L}(u(x; c), x)\f$
   */
  virtual Eigen::MatrixXd JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const;

  /// Evaluate the Jacobian of the operator with respect to the cofficients \f$c\f$, \f$\nabla_{c} \mathcal{L}(u(x; c), x)\f$ using finite difference
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  Eigen::MatrixXd JacobianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const;

  /// Compute the weighted sum of the Hessian of each output of the operator with respect to the cofficients \f$c\f$, \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     Defaults to approximating the Hessian with finite difference (see SystemOfEquations::HessianWRTCoefficientsFD)
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     @param[in] wieghts The weights for the sum
     \return The operator Hessian \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$
   */
  virtual Eigen::MatrixXd HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const;

  /// Compute the weighted sum of the Hessian using finite difference of each output of the operator with respect to the cofficients \f$c\f$, \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     @param[in] wieghts The weights for the sum
     \return The operator Hessian \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$
   */
  Eigen::MatrixXd HessianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coef, Eigen::VectorXd const& weights) const;

  /// The input dimension \f$d\f$
  const std::size_t indim;

  /// The output dimension \f$m\f$
  const std::size_t outdim;

protected:

  ///The parameters for this system of equations
  std::shared_ptr<const Parameters> para;

  /// The default value for the finite diference delta
  inline static double deltaFD_DEFAULT = 1.0e-2;

  /// The default value for the finite diference order
  inline static std::size_t orderFD_DEFAULT = 8;

private:
};

} // namespace clf 

#endif
