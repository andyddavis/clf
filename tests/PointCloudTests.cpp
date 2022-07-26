#include <gtest/gtest.h>

#include "clf/PointCloud.hpp"

using namespace clf;

TEST(PointCloudTests, DefaultConstruction) {
  std::size_t dim = 3;

  PointCloud cloud; // create an empty point cloud
  EXPECT_EQ(cloud.NumPoints(), 0);

  // create a random point 
  const Point pt(Eigen::VectorXd::Random(dim));
  cloud.AddPoint(pt);
  EXPECT_EQ(cloud.NumPoints(), 1);

  // create a random point in place
  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  cloud.AddPoint(x);
  EXPECT_EQ(cloud.NumPoints(), 2);

  EXPECT_NEAR((cloud.Get(0).x-pt.x).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((cloud.Get(1).x-x).norm(), 0.0, 1.0e-14);
}
