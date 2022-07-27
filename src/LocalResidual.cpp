#include "clf/LocalResidual.hpp"

#include <random>

using namespace clf;

LocalResidual::LocalResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, Point const& point, std::shared_ptr<const Parameters> const& para) :
  DensePenaltyFunction(func->NumCoefficients(), system->outdim*para->Get<std::size_t>("NumPoints"), para),
  points(GeneratePoints(point, para->Get<std::size_t>("NumPoints"), para->Get<double>("Radius"))),
  function(func),
  system(system)
{}

PointCloud LocalResidual::GeneratePoints(Point const& point, std::size_t const num, double const delta) {
  PointCloud points;
  std::normal_distribution gauss;
  std::default_random_engine generator;
  for( std::size_t i=0; i<num; ++i ) {
    // generate a point form a uniform distribution in the ball
    Eigen::VectorXd p(point.x.size());
    for( std::size_t j=0; j<point.x.size(); ++j ) { p(j) = gauss(generator); }
    const double u = rand()/(double)RAND_MAX;
    p *= delta*u/p.norm();

    // add the center point
    p += point.x;

    // add the point to the cloud
    points.AddPoint(p);
  }

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
