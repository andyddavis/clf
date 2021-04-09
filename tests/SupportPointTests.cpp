#include <gtest/gtest.h>

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(SupportPointTests, Construction) {
  constexpr std::size_t indim = 8, outdim = 4;
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  pt::ptree pt;
  pt.put("OutputDimension", outdim);

  SupportPoint point(x, pt);
  EXPECT_NEAR((point.x-x).norm(), 0.0, 1.0e-12);
  EXPECT_EQ(point.InputDimension(), indim);
  EXPECT_EQ(point.OutputDimension(), outdim);
}
