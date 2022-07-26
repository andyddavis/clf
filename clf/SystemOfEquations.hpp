#ifndef SYSTEMOFEQUATIONS_HPP_
#define SYSTEMOFEQUATIONS_HPP_

#include "clf/Parameters.hpp"

#include "clf/LocalFunction.hpp"

namespace clf {

/// A system of equations with the form \f$\mathcal{L}(u(x), x) - f(x) = 0\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$ and \f$u(x) = \Phi(x)^{\top} c\f$.
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
     Defaults to \f$\mathcal{L}(u(x), x) = u(x)\f$.
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const = 0;

  /// Evaluate the Jacobian of the operator with respect to the cofficients \f$c\f$, \f$\nabla_{c} \mathcal{L}(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     Defaults to approximating the Jacobian with finite difference (see SystemOfEquations::JacobianWRTCoefficientsFD)
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
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

  /// The input dimension \f$d\f$
  const std::size_t indim;

  /// The output dimension \f$m\f$
  const std::size_t outdim;

private:

  ///The parameters for this system of equations
  std::shared_ptr<const Parameters> para;

  /// The default value for the finite diference delta
  inline static double deltaFD_DEFAULT = 1.0e-2;

  /// The default value for the finite diference order
  inline static std::size_t orderFD_DEFAULT = 8;
};

} // namespace clf 

#endif
