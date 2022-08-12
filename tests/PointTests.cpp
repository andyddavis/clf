#include <gtest/gtest.h>

#include "clf/Point.hpp"

using namespace clf;

TEST(PointTests, BasicTest) {
  // create a random point of dimension dim
  const std::size_t dim = 5;


  Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  const Point point0(x);
  EXPECT_NEAR((x-point0.x).norm(), 0.0, 1.0e-14);

  x = Eigen::VectorXd::Random(dim);
  const Point point1(x);
  EXPECT_NEAR((x-point1.x).norm(), 0.0, 1.0e-14);
  EXPECT_NE(point0.id, point1.id);
}

