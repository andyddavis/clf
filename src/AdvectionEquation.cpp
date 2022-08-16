#include "clf/AdvectionEquation.hpp"

using namespace clf;

AdvectionEquation::AdvectionEquation(std::size_t const indim, double const constantVel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(indim, para), constantVel(constantVel)
{}

AdvectionEquation::AdvectionEquation(Eigen::VectorXd const& vel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(vel.size(), para), vel(vel)
{}

Eigen::VectorXd AdvectionEquation::Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==1);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);
  if( constantVel ) { return Eigen::VectorXd::Constant(indim, (*constantVel) * u->Evaluate(x, coeff) [0]); }
  assert(vel);
  return (*vel) * u->Evaluate(x, coeff) [0];
}

double AdvectionEquation::FluxDivergence(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  return 0.0;
}
