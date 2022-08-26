#include "clf/Residual.hpp"

using namespace clf;

Residual::Residual(std::shared_ptr<PointCloud> const& cloud, std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para) :
  DensePenaltyFunction(func->NumCoefficients(), system->outdim*cloud->NumPoints(), para),
  cloud(cloud),
  function(func),
  system(system)
{}

std::size_t Residual::NumPoints() const { return cloud->NumPoints(); }

std::shared_ptr<Point> Residual::GetPoint(std::size_t const ind) const { return cloud->Get(ind); }

Eigen::VectorXd Residual::Evaluate(Eigen::VectorXd const& beta) {
  Eigen::VectorXd resid(outdim);
  std::size_t start = 0;
  for( std::size_t i=0; i<NumPoints(); ++i ) {
    const Eigen::VectorXd& x = cloud->Get(i)->x;
    resid.segment(start, system->outdim) = system->Operator(function, x, beta) - system->RightHandSide(x);
    start += system->outdim;
  }

  return resid;
}

Eigen::MatrixXd Residual::Jacobian(Eigen::VectorXd const& beta) {
  Eigen::MatrixXd jac(outdim, indim);
  std::size_t start = 0;
  for( std::size_t i=0; i<NumPoints(); ++i ) {
    const Eigen::VectorXd& x = cloud->Get(i)->x;
    jac.block(start, 0, system->outdim, indim) = system->JacobianWRTCoefficients(function, x, beta);
    start += system->outdim;
  }

  return jac;
}

Eigen::MatrixXd Residual::Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(indim, indim);

  std::size_t start = 0;
  for( std::size_t i=0; i<NumPoints(); ++i ) {
    hess += system->HessianWRTCoefficients(function, cloud->Get(i)->x, beta, weights.segment(start, system->outdim));
    start += system->outdim;
  }

  return hess;
}

std::size_t Residual::SystemID() const { return system->id; }
