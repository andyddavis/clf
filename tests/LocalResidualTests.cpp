#include <gtest/gtest.h>

#include "clf/LocalResidual.hpp"

using namespace clf;

TEST(LocalResidualTests, LinearSystem) {
  const double radius = 0.1;
  const std::size_t numPoints = 100;
  const std::size_t dim = 4;
  const Eigen::VectorXd point = Eigen::VectorXd::Random(dim);

  auto para = std::make_shared<Parameters>();
  para->Add<std::size_t>("NumPoints", numPoints);
  para->Add<double>("Radius", radius);

  LocalResidual resid(point, para);
  EXPECT_EQ(resid.NumLocalPoints(), numPoints);
}
