#include "clf/BoundaryCondition.hpp"

using namespace clf;

BoundaryCondition::BoundaryCondition(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para) :
  Residual(std::make_shared<PointCloud>(), func, system, para)
{}

void BoundaryCondition::AddPoint(std::pair<Eigen::VectorXd, Eigen::VectorXd> const& pnt) {
  cloud->AddPoint(std::make_shared<Point>(pnt));
  outdim += system->outdim;
}
