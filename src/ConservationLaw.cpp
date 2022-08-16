#include "clf/ConservationLaw.hpp"

#include "clf/FiniteDifference.hpp"

using namespace clf;

ConservationLaw::ConservationLaw(std::size_t const indim, std::shared_ptr<const Parameters> const& para) :
  SystemOfEquations(indim, 1, para)
{
  assert(para);
}

Eigen::VectorXd ConservationLaw::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  return Eigen::VectorXd::Constant(1, FluxDivergence(u, x, coeff));
}

double ConservationLaw::FluxDivergence(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const { return FluxDivergenceFD(u, x, coeff); }

double ConservationLaw::FluxDivergenceFD(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(para);
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd xcopy = x; // need to copy to get ride of const
  double div = 0.0;
  for( std::size_t i=0; i<indim; ++i ) {
    div += FiniteDifference::Derivative<double>(i, delta, weights, xcopy, [this, &i, &u, &coeff](Eigen::VectorXd const& x) { return this->Flux(u, x, coeff) [i]; });
  }
  
  return div;
}
