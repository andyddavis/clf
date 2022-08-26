#include "clf/SystemOfEquations.hpp"

#include "clf/CLFExceptions.hpp"

#include "clf/FiniteDifference.hpp"

using namespace clf;

std::atomic<std::size_t> SystemOfEquations::nextID = 0;

SystemOfEquations::SystemOfEquations(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para) :
  indim(indim), outdim(outdim), id(nextID++),
  para(para)
{
  assert(para);
}

SystemOfEquations::SystemOfEquations(std::shared_ptr<const Parameters> const& para) :
  indim(para->Get<std::size_t>("InputDimension")), outdim(para->Get<std::size_t>("OutputDimension")), id(nextID++),
  para(para)
{}

Eigen::VectorXd SystemOfEquations::RightHandSide(Eigen::VectorXd const& x) const { return Eigen::VectorXd::Zero(outdim); }

Eigen::VectorXd SystemOfEquations::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  throw exceptions::NotImplemented("SystemOfEquations::Operator");
  return Eigen::VectorXd();
}

Eigen::MatrixXd SystemOfEquations::JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { return JacobianWRTCoefficientsFD(u, x, coeff); }

Eigen::MatrixXd SystemOfEquations::JacobianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::MatrixXd jac(outdim, u->NumCoefficients());
  Eigen::VectorXd c = coeff;
  for( std::size_t i=0; i<u->NumCoefficients(); ++i ) { jac.col(i) = FiniteDifference::Derivative<Eigen::VectorXd>(i, delta, weights, c, [this, &u, &x](Eigen::VectorXd const& c) { return this->Operator(u, x, c); }); }

  return jac;
}

Eigen::MatrixXd SystemOfEquations::HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const { return HessianWRTCoefficientsFD(u, x, coeff, weights); }

Eigen::MatrixXd SystemOfEquations::HessianWRTCoefficientsFD(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& sumWeights) const {
  assert(sumWeights.size()==outdim);

  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(coeff.size(), coeff.size());

  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd c = coeff;
  for( std::size_t i=0; i<u->NumCoefficients(); ++i ) {
    const Eigen::MatrixXd secondDeriv = FiniteDifference::Derivative<Eigen::MatrixXd>(i, delta, weights, c, [this, &u, &x](Eigen::VectorXd const& c) { return this->JacobianWRTCoefficients(u, x, c); });

    for( std::size_t j=0; j<outdim; ++j ) { hess.row(i) += sumWeights(j)*secondDeriv.row(j); }
  }

  return hess;
}
