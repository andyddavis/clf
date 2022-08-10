#include "clf/LocalResidual.hpp"

using namespace clf;

LocalResidual::LocalResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para) :
  DensePenaltyFunction(func->NumCoefficients(), system->outdim*para->Get<std::size_t>("NumPoints"), para),
  points(GeneratePoints(func, para->Get<std::size_t>("NumPoints"))),
  function(func),
  system(system)
{}

PointCloud LocalResidual::GeneratePoints(std::shared_ptr<LocalFunction> const& func, std::size_t const num) {
  PointCloud points;
  for( std::size_t i=0; i<num; ++i ) { points.AddPoint(func->SampleDomain()); }
  return points;
}

std::size_t LocalResidual::NumLocalPoints() const { return points.NumPoints(); }

Eigen::VectorXd LocalResidual::Evaluate(Eigen::VectorXd const& beta) {
  Eigen::VectorXd resid(outdim);
  std::size_t start = 0;
  for( std::size_t i=0; i<NumLocalPoints(); ++i ) {
    const Eigen::VectorXd& x = points.Get(i).x;
    resid.segment(start, system->outdim) = system->Operator(function, x, beta) - system->RightHandSide(x);
    start += system->outdim;
  }

  return resid;
}

Point LocalResidual::GetPoint(std::size_t const ind) const { return points.Get(ind); }

Eigen::MatrixXd LocalResidual::Jacobian(Eigen::VectorXd const& beta) {
  Eigen::MatrixXd jac(outdim, indim);
  std::size_t start = 0;
  for( std::size_t i=0; i<NumLocalPoints(); ++i ) {
    const Eigen::VectorXd& x = points.Get(i).x;
    jac.block(start, 0, system->outdim, indim) = system->JacobianWRTCoefficients(function, x, beta);
    start += system->outdim;
  }

  return jac;
}

Eigen::MatrixXd LocalResidual::Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(indim, indim);

  std::size_t start = 0;
  for( std::size_t i=0; i<NumLocalPoints(); ++i ) {
    hess += system->HessianWRTCoefficients(function, points.Get(i).x, beta, weights.segment(start, system->outdim));
    start += system->outdim;
  }

  return hess;
}
