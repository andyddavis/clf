#include "clf/ConservationLaw.hpp"

#include "clf/FiniteDifference.hpp"

using namespace clf;

ConservationLaw::ConservationLaw(std::size_t const indim, std::shared_ptr<const Parameters> const& para) :
  SystemOfEquations(indim, 1, para)
{
  assert(para);
}

Eigen::VectorXd ConservationLaw::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { return Eigen::VectorXd::Constant(1, FluxDivergence(u, x, coeff)); }

Eigen::MatrixXd ConservationLaw::JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { return FluxDivergence_GradientWRTCoefficients(u, x, coeff).transpose(); }

Eigen::MatrixXd ConservationLaw::HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const { return weights(0)*FluxDivergence_HessianWRTCoefficients(u, x, coeff); }

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

Eigen::VectorXd ConservationLaw::FluxDivergence_GradientWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const { return FluxDivergence_GradientWRTCoefficientsFD(u, x, coeff); }

Eigen::VectorXd ConservationLaw::FluxDivergence_GradientWRTCoefficientsFD(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd deriv(coeff.size());
  Eigen::VectorXd coeffCopy = coeff; // need to copy to get ride of const
  for( std::size_t i=0; i<coeff.size(); ++i ) {
    deriv(i) = FiniteDifference::Derivative<double>(i, delta, weights, coeffCopy, [this, &u, &x](Eigen::VectorXd const& coeff) { return this->FluxDivergence(u, x, coeff); });
  }

  return deriv;
}

Eigen::MatrixXd ConservationLaw::FluxDivergence_HessianWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const { return FluxDivergence_HessianWRTCoefficientsFD(u, x, coeff); }

Eigen::MatrixXd  ConservationLaw::FluxDivergence_HessianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));
  
  Eigen::MatrixXd hess(coeff.size(), coeff.size());
  Eigen::VectorXd coeffCopy = coeff; // need to copy to get ride of const
  for( std::size_t i=0; i<coeff.size(); ++i ) {
    hess.col(i) = FiniteDifference::Derivative<Eigen::VectorXd>(i, delta, weights, coeffCopy, [this, &u, &x](Eigen::VectorXd const& coeff) { return this->FluxDivergence_GradientWRTCoefficients(u, x, coeff); });
  }

  return hess;
}

Eigen::MatrixXd ConservationLaw::Flux_JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const { return Flux_JacobianWRTCoefficientsFD(u, x, coeff); }

Eigen::MatrixXd ConservationLaw::Flux_JacobianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::MatrixXd jac(indim, coeff.size());
  Eigen::VectorXd coeffCopy = coeff; // need to copy to get ride of const
  for( std::size_t i=0; i<coeff.size(); ++i ) {
    jac.col(i) = FiniteDifference::Derivative<Eigen::VectorXd>(i, delta, weights, coeffCopy, [this, &u, &x](Eigen::VectorXd const& coeff) { return this->Flux(u, x, coeff); });
  }
  
  return jac;
}

Eigen::MatrixXd ConservationLaw::Flux_HessianWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff, Eigen::VectorXd const& weights) const { return Flux_HessianWRTCoefficientsFD(u, x, coeff, weights); }

Eigen::MatrixXd ConservationLaw::Flux_HessianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff, Eigen::VectorXd const& weights) const {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weightsFD = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(coeff.size(), coeff.size());
  Eigen::VectorXd coeffCopy = coeff; // need to copy to get ride of const
  for( std::size_t i=0; i<coeff.size(); ++i ) {
    hess.row(i) = ( FiniteDifference::Derivative<Eigen::MatrixXd>(i, delta, weightsFD, coeffCopy, [this, &u, &x](Eigen::VectorXd const& coeff) { return this->Flux_JacobianWRTCoefficients(u, x, coeff); }).array().colwise()*weights.array() ).matrix().colwise().sum();
  }
  
  return hess;
}
