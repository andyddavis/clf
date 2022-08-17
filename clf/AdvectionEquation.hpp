#ifndef ADVECTIONEQUATION_HPP_
#define ADVECTIONEQUATION_HPP_

#include "clf/ConservationLaw.hpp"

namespace clf {

/// The advection equation \f$\nabla \cdot (u \boldsymbol{v}) - f = 0\f$
class AdvectionEquation : public ConservationLaw {
public:
  
  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] constantVel The constant velocity \f$v_i = v\f$ (defaults to \f$1\f$)
     @param[in] para The parameters for this system of equations
   */
  AdvectionEquation(std::size_t const indim, double const constantVel = 1.0, std::shared_ptr<const Parameters> const& para = std::make_shared<const Parameters>());

  /**
     @param[in] vel The velocity vector \f$\boldsymbol{v}\f$
     @param[in] para The parameters for this system of equations
   */
  AdvectionEquation(Eigen::VectorXd const& vel, std::shared_ptr<const Parameters> const& para = std::make_shared<const Parameters>());

virtual ~AdvectionEquation() = default;

  /// Compute the flux \f$F(u(\cdot), x)\f$
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The flux \f$F(u(\cdot), x)\f$
   */
  virtual Eigen::VectorXd Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const final override;

  /// Compute the divergence of the flux \f$\nabla \cdot F(u(\cdot), x)\f$
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The flux divergence \f$\nabla \cdot F(u(\cdot), x)\f$
  */
  virtual double FluxDivergence(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const final override;

  /// Compute the gradient of the flux divergence \f$\nabla_{c} ( \nabla_x \cdot F(u(\cdot), x) )\f$ with respect to the coefficients
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The gradient of the flux divergence  \f$\nabla_{c} ( \nabla_x \cdot F(u(\cdot), x) )\f$ with respect to the coefficients
   */
  virtual Eigen::VectorXd FluxDivergence_GradientWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const final override;

  /// Compute the Hessian of the flux divergence \f$H_{c} ( \nabla_x \cdot F(u(\cdot), x) )\f$ with respect to the coefficients
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The Hessian of the flux divergence \f$H_{c} ( \nabla_x \cdot F(u(\cdot), x) )\f$ with respect to the coefficients
   */
  virtual Eigen::MatrixXd FluxDivergence_HessianWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const final override;
  
private:

  /// The linear differential operator that takes the first derivative of the local function with respect to each input coordinate
  std::shared_ptr<LinearDifferentialOperator> linOper = std::make_shared<LinearDifferentialOperator>(Eigen::MatrixXi::Identity(indim, indim), 1);

  /// Store the velocity \f$v_i = v\f$ as a constant
  std::optional<double> constantVel;

  /// The velocity vector \f$\boldsymbol{v}\f$
  std::optional<Eigen::VectorXd> vel;
};
  
} // namespace clf

#endif
