#include "clf/BurgersEquation.hpp"

using namespace clf;

BurgersEquation::BurgersEquation(std::size_t const indim, double const constantVel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(indim, para), constantVel(constantVel)
{}

BurgersEquation::BurgersEquation(Eigen::VectorXd const& vel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(vel.size(), para), vel(vel)
{}


Eigen::VectorXd BurgersEquation::Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==1);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);
  const double eval = u->Evaluate(x, coeff) [0];
  if( constantVel ) { return Eigen::VectorXd::Constant(indim, (*constantVel) * 0.5*eval*eval); }
  assert(vel);
  return (*vel) * 0.5*eval*eval;
}
