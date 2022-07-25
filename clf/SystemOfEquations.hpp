#ifndef SYSTEMOFEQUATIONS_HPP_
#define SYSTEMOFEQUATIONS_HPP_

#include "clf/LocalFunction.hpp"

namespace clf {

/// A system of equations with the form \f$\mathcal{L}(u(x), x) - f(x) = 0\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$ and \f$u(x) = \Phi(x)^{\top} c\f$.
/**
   Let \f$u: \Omega \mapsto \mathbb{R}^{m}\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$. The system of equations is composed of two parts. First, \f$\mathcal{L}(u(x), x)\f$ is the operator and, second, \f$f(x)\f$ is the right hand side.
 */
class SystemOfEquations {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$m\f$
   */
  SystemOfEquations(std::size_t const indim, std::size_t const outdim);
  
  virtual ~SystemOfEquations() = default;

  /// Evaluate the right hand side at the location \f$x\f$.
  /**
     Defaults to \f$f(x)=0\f$.
     @param[in] x The location \f$x\f$
     \return The right hand side \f$f(x)\f$
   */
  virtual Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const;

  /// Evaluate the operator given the function \f$u\f$ and at the location \f$x\f$.
  /**
     Defaults to \f$\mathcal{L}(u(x), x) = u(x)\f$.
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const;

  /// The input dimension \f$d\f$
  const std::size_t indim;

  /// The output dimension \f$m\f$
  const std::size_t outdim;

private:
};

} // namespace clf 

#endif
