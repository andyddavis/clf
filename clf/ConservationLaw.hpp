#ifndef CONSERVATIONLAW_HPP_
#define CONSERVATIONLAW_HPP_

#include "clf/SystemOfEquations.hpp"

namespace clf {


/// A system of equations with the form \f$\nabla \cdot F(u(\cdot), x) - f(x) = 0\f$ with \f$x \in \Omega \subseteq \mathbb{R}^{d}\f$, \f$u: \Omega \mapsto \mathbb{R}\f$, and \f$F(u(\cdot), x) \in \mathbb{R}^{d}\f$
/**
Let \f$F(u(\cdot), x)\f$ be the flux. For example, setting \f$F_i(u(\cdot), [t, x]) = v_i u\f$, where \f$v_i \in \mathbb{R}\f$, defines the advection equation \f$\frac{\partial}{\partial t} u + \sum_{i=1}^{d} v_i \frac{\partial}{\partial x_i} u - f = 0\f$ (see clf::AdvectionEquation). Additionally, defining \f$F(u(\cdot), [t, x]) = [u, \frac{1}{2} v_1 u^2,\, ...,\, \frac{1}{2} v_d u^2]\f$ defines Burgers' equation \f$\frac{\partial}{\partial t} u + \frac{1}{2} \sum_{i=1}^{d} v_i \frac{\partial}{\partial x_i} u^2 - f = 0\f$ (see clf::BurgersEquation).
*/
class ConservationLaw : public SystemOfEquations{
public:
  
  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] para The parameters for this system of equations
   */
  ConservationLaw(std::size_t const indim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~ConservationLaw() = default;

  /// Compute the flux \f$F(u(\cdot), x)\f$
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The flux \f$F(u(\cdot), x)\f$
   */
  virtual Eigen::VectorXd Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const = 0;

  /// Compute the divergence of the flux \f$\nabla \cdot F(u(\cdot), x)\f$
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The flux divergence \f$\nabla \cdot F(u(\cdot), x)\f$
   */
  virtual double FluxDivergence(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const;

  /// Compute the divergence of the flux \f$\nabla \cdot F(u(\cdot), x)\f$ using finite difference
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The flux divergence \f$\nabla \cdot F(u(\cdot), x)\f$
   */
  double FluxDivergenceFD(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const;

  /// Evaluate the operator \f$\nabla \cdot F(u(\cdot), x)\f$. given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the local function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const final override;
  
private:
};
  
} // namespace clf

#endif
