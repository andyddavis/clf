#include "clf/DensePenaltyFunction.hpp"

using namespace clf;

DensePenaltyFunction::DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para) :
  PenaltyFunction<Eigen::MatrixXd>(indim, outdim, para)
{}

Eigen::MatrixXd DensePenaltyFunction::JacobianFD(Eigen::VectorXd const& beta) {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));
  
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outdim, indim);
  Eigen::VectorXd b = beta;
  for( std::size_t i=0; i<indim; ++i ) { jac.col(i) = FirstDerivativeFD(i, delta, weights, b); }
  return jac;
}

Eigen::MatrixXd DensePenaltyFunction::HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& sumWeights) {
  assert(sumWeights.size()==outdim);
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(indim, indim);

  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd b = beta;
  for( std::size_t i=0; i<indim; ++i ) {
    const Eigen::MatrixXd secondDeriv = FiniteDifference::Derivative<Eigen::MatrixXd>(i, delta, weights, b, [this](Eigen::VectorXd const& beta) { return this->Jacobian(beta); });
    
    for( std::size_t j=0; j<outdim; ++j ) { hess.row(i) += sumWeights(j)*secondDeriv.row(j); }
  }

  return hess;
}
