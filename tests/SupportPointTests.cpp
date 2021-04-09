#include <gtest/gtest.h>

#include "clf/SupportPoint.hpp"

using namespace clf;

TEST(SupportPointTests, Construction) {
  constexpr std::size_t dim = 8;
  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

  SupportPoint point(x);
  EXPECT_NEAR((point.x-x).norm(), 0.0, 1.0e-12);
}
