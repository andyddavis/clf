#include "clf/LocalResidual.hpp"

using namespace clf;

LocalResidual::LocalResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para) :
  Residual(GeneratePoints(func, para->Get<std::size_t>("NumPoints")), func, system, para)
{}

std::shared_ptr<PointCloud> LocalResidual::GeneratePoints(std::shared_ptr<LocalFunction> const& func, std::size_t const num) {
  auto points = std::make_shared<PointCloud>(func->GetDomain());
  points->AddPoints(num);
  return points;
}

