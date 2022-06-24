#include <gtest/gtest.h>

#include "clf/Point.hpp"

using namespace clf;

TEST(PointTests, BasicTest) {
  // create a random point of dimension dim
  const std::size_t dim = 5;
  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

  const Point point(x);
  EXPECT_NEAR((x-point.x).norm(), 0.0, 1.0e-14);
}

