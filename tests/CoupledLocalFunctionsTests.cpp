#include <gtest/gtest.h>

#include "clf/CoupledLocalFunctions.hpp"

using namespace clf;

TEST(CoupledLocalFunctionsTests, EvaluationTests) {
  const std::size_t indim = 5;
  const std::size_t numPoints = 10;
  
  auto cloud = std::make_shared<PointCloud>();
  for( std::size_t i=0; i<numPoints; ++i ) { cloud->AddPoint(Eigen::VectorXd::Random(indim)); }

  CoupledLocalFunctions func(cloud);
  EXPECT_EQ(func.NumLocalFunctions(), numPoints);
}
