#include "clf/SystemOfEquations.hpp"

#include "clf/CLFExceptions.hpp"

#include "clf/FiniteDifference.hpp"

using namespace clf;

SystemOfEquations::SystemOfEquations(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para) :
  indim(indim), outdim(outdim),
  para(para)
{}

Eigen::VectorXd SystemOfEquations::RightHandSide(Eigen::VectorXd const& x) const { return Eigen::VectorXd::Zero(outdim); }

Eigen::MatrixXd SystemOfEquations::JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { return JacobianWRTCoefficients(u, x, coeff); }

Eigen::MatrixXd SystemOfEquations::JacobianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::MatrixXd jac(outdim, u->NumCoefficients());
  Eigen::VectorXd c = coeff;
  for( std::size_t i=0; i<u->NumCoefficients(); ++i ) { jac.col(i) = FiniteDifference::Derivative<Eigen::VectorXd>(i, delta, weights, c, [this, &u, &x](Eigen::VectorXd const& c) { return this->Operator(u, x, c); }); }

  return jac;
}
