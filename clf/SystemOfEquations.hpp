#ifndef SYSTEMOFEQUATIONS_HPP_
#define SYSTEMOFEQUATIONS_HPP_

#include "clf/LocalFunction.hpp"

namespace clf {

/// A system of equations with the form \f$\mathcal{L}(u(x), x) - f(x) = 0\f$, where \f$u(x) = \Phi(x)^{\top} c\f$.
/**
   Let \f$u: \Omega \mapsto \mathbb{R}^{m}\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$. The system of equations is composed of two parts. First, \f$\mathcal{L}(u(x), x)\f$ is the operator and, second, \f$f(x)\f$ is the right hand side.
 */
class SystemOfEquations {
public:

  SystemOfEquations();
  
  virtual ~SystemOfEquations() = default;

  /// Evaluate the right hand side at the location \f$x\f$.
  /**
     @param[in] x The location \f$x\f$
     \return The right hand side \f$f(x)\f$
   */
  virtual Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const = 0;

  /// Evaluate the operator given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x) const = 0;

private:
};

} // namespace clf 

#endif
