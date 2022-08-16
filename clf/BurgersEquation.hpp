#ifndef BURGERSEQUATION_HPP_
#define BURGERSEQUATION_HPP_

#include "clf/ConservationLaw.hpp"

namespace clf {
  
/// Burgers' equation \f$\frac{\partial}{\partial t} + \frac{1}{2} \nabla \cdot (u^2 \boldsymbol{v}) - f = 0\f$
class BurgersEquation : public ConservationLaw {
public:
  
  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] constantVel The constant velocity \f$v_i = v\f$ (defaults to \f$1\f$)
     @param[in] para The parameters for this system of equations
   */
  BurgersEquation(std::size_t const indim, double const constantVel = 1.0, std::shared_ptr<const Parameters> const& para = std::make_shared<const Parameters>());

  /**
     @param[in] vel The velocity vector \f$\boldsymbol{v}\f$
     @param[in] para The parameters for this system of equations
   */
  BurgersEquation(Eigen::VectorXd const& vel, std::shared_ptr<const Parameters> const& para = std::make_shared<const Parameters>());

  virtual ~BurgersEquation() = default;

  /// Compute the flux \f$F(u(\cdot), x)\f$
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The flux \f$F(u(\cdot), x)\f$
   */
  virtual Eigen::VectorXd Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const override;

private:
  
  /// Store the velocity \f$v_i = v\f$ as a constant
  std::optional<double> constantVel;

  /// The velocity vector \f$\boldsymbol{v}\f$
  std::optional<Eigen::VectorXd> vel;
};
  
} // namespace clf

#endif
