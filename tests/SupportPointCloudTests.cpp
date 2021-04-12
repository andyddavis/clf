#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/RandomVariable.h>
#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

class SupportPointCloudTests : public::testing::Test {
protected:
  /// Set up information to test the support point cloud
  virtual void SetUp() override {
    ptSupportPoints.put("OutputDimension", outdim);
    ptSupportPoints.put("BasisFunctions.Type", "TotalOrderPolynomials");
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {
  }

  /// The domain dimension
  const std::size_t indim = 8;

  /// The output dimension
  const std::size_t outdim = 4;

  /// The number of support points
  const std::size_t n = 100;

  /// Options for the support points
  pt::ptree ptSupportPoints;

  /// Options for the support point cloud
  pt::ptree ptSupportPointCloud;
};

TEST_F(SupportPointCloudTests, Construction) {
  // create a bunch of random support points
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(n);
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  for( std::size_t i=0; i<n; ++i ) { supportPoints[i] = std::make_shared<SupportPoint>(dist->Sample(), ptSupportPoints); }

  // create the support point cloud
  SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  EXPECT_EQ(cloud.NumSupportPoints(), n);
  EXPECT_EQ(cloud.kdtree_get_point_count(), n);
  EXPECT_EQ(cloud.InputDimension(), indim);
  EXPECT_EQ(cloud.OutputDimension(), outdim);
}

TEST(SupportPointCloudErrorTests, InputDimensionCheck) {
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(2);

  // create two points with different input sizes
  pt::ptree ptSupportPoints;
  ptSupportPoints.put("BasisFunctions.Type", "TotalOrderPolynomials");
  supportPoints[0] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(3), ptSupportPoints);
  supportPoints[1] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(5), ptSupportPoints);

  // try to create a support point cloud
  try {
    pt::ptree ptSupportPointCloud;
    SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  } catch( SupportPointCloudDimensionException const& exc ) {
    EXPECT_EQ(exc.type, SupportPointCloudDimensionException::Type::INPUT);
    EXPECT_NE(exc.ind1, exc.ind2);
    EXPECT_NE(supportPoints[exc.ind1]->InputDimension(), supportPoints[exc.ind2]->InputDimension());
  }
}

TEST(SupportPointCloudErrorTests, OutputDimensionCheck) {
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(2);

  // create two points with different input sizes
  pt::ptree ptSupportPoints;
  ptSupportPoints.put("BasisFunctions.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("OutputDimension", 2);
  supportPoints[0] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(5), ptSupportPoints);
  ptSupportPoints.put("OutputDimension", 8);
  supportPoints[1] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(5), ptSupportPoints);

  // try to create a support point cloud
  try {
    pt::ptree ptSupportPointCloud;
    SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  } catch( SupportPointCloudDimensionException const& exc ) {
    EXPECT_EQ(exc.type, SupportPointCloudDimensionException::Type::OUTPUT);
    EXPECT_NE(exc.ind1, exc.ind2);
    EXPECT_NE(supportPoints[exc.ind1]->OutputDimension(), supportPoints[exc.ind2]->OutputDimension());
  }
}
