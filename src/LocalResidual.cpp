#include "clf/LocalResidual.hpp"

#include <random>

using namespace clf;

LocalResidual::LocalResidual(Eigen::VectorXd const& point, std::shared_ptr<const Parameters> const& para) :
  DensePenaltyFunction(1, 1, para),
  points(GeneratePoints(point, para->Get<std::size_t>("NumPoints"), para->Get<double>("Radius")))
{}

std::vector<Eigen::VectorXd> LocalResidual::GeneratePoints(Eigen::VectorXd const& point, std::size_t const num, double const delta) {
  std::vector<Eigen::VectorXd> points(num);
  std::normal_distribution gauss;
  std::default_random_engine generator;
  for( std::size_t i=0; i<num; ++i ) {
    // generate a point form a uniform distribution in the ball
    points[i].resize(point.size());
    for( std::size_t j=0; j<point.size(); ++j ) { points[i](j) = gauss(generator); }
    const double u = rand()/(double)RAND_MAX;
    points[i] *= delta*u/points[i].norm();

    // add the center point
    points[i] += point;
  }

  return points;
}

std::size_t LocalResidual::NumLocalPoints() const { return points.size(); }

Eigen::VectorXd LocalResidual::Evaluate(Eigen::VectorXd const& beta) {
  return Eigen::VectorXd();
}
