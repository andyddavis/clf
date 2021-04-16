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
    ptSupportPoints.put("BasisFunctions", "Basis1, Basis2, Basis3, Basis4");
    ptSupportPoints.put("Basis1.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis2.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis3.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis4.Type", "TotalOrderPolynomials");
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {
    EXPECT_EQ(cloud->NumSupportPoints(), n);
    EXPECT_EQ(cloud->kdtree_get_point_count(), n);
    EXPECT_EQ(cloud->InputDimension(), indim);
    EXPECT_EQ(cloud->OutputDimension(), outdim);
  }

  /// The domain dimension
  const std::size_t indim = 6;

  /// The output dimension
  const std::size_t outdim = 4;

  /// The number of support points
  const std::size_t n = 50;

  /// Options for the support points
  pt::ptree ptSupportPoints;

  /// Options for the support point cloud
  pt::ptree ptSupportPointCloud;

  /// The support point cloud
  std::shared_ptr<SupportPointCloud> cloud;
};

TEST_F(SupportPointCloudTests, Construction) {
  // create a bunch of random support points
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(n);
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  for( std::size_t i=0; i<n; ++i ) { supportPoints[i] = std::make_shared<SupportPoint>(dist->Sample(), ptSupportPoints); }

  // create the support point cloud
  cloud = std::make_shared<SupportPointCloud>(supportPoints, ptSupportPointCloud);
}

TEST_F(SupportPointCloudTests, NearestNeighborSearch) {
  // create a bunch of random support points
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(n);
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  for( std::size_t i=0; i<n; ++i ) { supportPoints[i] = std::make_shared<SupportPoint>(dist->Sample(), ptSupportPoints); }

  // create the support point cloud
  cloud = std::make_shared<SupportPointCloud>(supportPoints, ptSupportPointCloud);

  // find 5 nearest neighbors
  std::vector<std::size_t> neighInd;
  std::vector<double> neighDist;
  const Eigen::VectorXd newpoint = dist->Sample();
  cloud->FindNearestNeighbors(newpoint, 5, neighInd, neighDist);

  // get the maximum squared distance
  const double maxit = *std::max_element(neighDist.begin(), neighDist.end());

  // loop through all of the support points
  std::size_t count = 0;
  for( std::size_t i=0; i<cloud->NumSupportPoints(); ++i ) {
    // try to find this point in the nearest neighbor lest
    auto ind = std::find(neighInd.begin(), neighInd.end(), i);

    const Eigen::VectorXd diff = cloud->GetSupportPoint(i)->x-newpoint;
    const double dist = diff.dot(diff);
    if( dist>maxit+1.0e-12 ) {
      EXPECT_TRUE(ind==neighInd.end());
    } else {
      EXPECT_TRUE(ind!=neighInd.end());

      const std::size_t index = ind-neighInd.begin();
      EXPECT_NEAR(dist, neighDist[index], 1.0e-12);

      ++count;
    }
  }

  // the number of nearest neighbors should be 5
  EXPECT_EQ(count, 5);
}

TEST(SupportPointCloudErrorTests, InputDimensionCheck) {
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(2);

  // create two points with different input sizes
  pt::ptree ptSupportPoints;
  ptSupportPoints.put("BasisFunctions", "Basis");
  ptSupportPoints.put("Basis.Type", "TotalOrderPolynomials");
  supportPoints[0] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(3), ptSupportPoints);
  supportPoints[1] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(5), ptSupportPoints);

  // try to create a support point cloud
  try {
    pt::ptree ptSupportPointCloud;
    SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  } catch( exceptions::SupportPointCloudDimensionException const& exc ) {
    EXPECT_EQ(exc.type, exceptions::SupportPointCloudDimensionException::Type::INPUT);
    EXPECT_NE(exc.ind1, exc.ind2);
    EXPECT_NE(supportPoints[exc.ind1]->InputDimension(), supportPoints[exc.ind2]->InputDimension());
  }
}

TEST(SupportPointCloudErrorTests, OutputDimensionCheck) {
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(2);

  // create two points with different input sizes
  pt::ptree ptSupportPoints;
  ptSupportPoints.put("OutputDimension", 2);
  ptSupportPoints.put("BasisFunctions", "Basis1, Basis2");
  ptSupportPoints.put("Basis1.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis2.Type", "TotalOrderPolynomials");
  supportPoints[0] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(5), ptSupportPoints);
  ptSupportPoints.put("OutputDimension", 8);
  ptSupportPoints.put("BasisFunctions", "Basis1, Basis2, Basis3, Basis4, Basis5, Basis6, Basis7, Basis8");
  ptSupportPoints.put("Basis1.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis2.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis3.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis4.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis5.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis6.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis7.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis8.Type", "TotalOrderPolynomials");
  supportPoints[1] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(5), ptSupportPoints);

  // try to create a support point cloud
  try {
    pt::ptree ptSupportPointCloud;
    SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  } catch( exceptions::SupportPointCloudDimensionException const& exc ) {
    EXPECT_EQ(exc.type, exceptions::SupportPointCloudDimensionException::Type::OUTPUT);
    EXPECT_NE(exc.ind1, exc.ind2);
    EXPECT_NE(supportPoints[exc.ind1]->OutputDimension(), supportPoints[exc.ind2]->OutputDimension());
  }
}

TEST(SupportPointCloudErrorTests, NotEnoughPoints) {
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(2);

  // create two points with different input sizes
  pt::ptree ptSupportPoints;
  ptSupportPoints.put("BasisFunctions", "Basis");
  ptSupportPoints.put("Basis.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("NumNeighbors", 10);
  supportPoints[0] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(3), ptSupportPoints);
  supportPoints[1] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(3), ptSupportPoints);

  // try to create a support point cloud
  try {
    pt::ptree ptSupportPointCloud;
    SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  } catch( exceptions::SupportPointCloudNotEnoughPointsException const& exc ) {
    EXPECT_EQ(exc.numPoints, supportPoints.size());
    EXPECT_EQ(exc.required, 10);
  }
}

TEST(SupportPointCloudErrorTests, NotConnected) {
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(4);

  // create two points with different input sizes
  pt::ptree ptSupportPoints;
  ptSupportPoints.put("BasisFunctions", "Basis");
  ptSupportPoints.put("Basis.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis.Order", 0);
  ptSupportPoints.put("NumNeighbors", 2);
  supportPoints[0] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(3), ptSupportPoints);
  supportPoints[1] = std::make_shared<SupportPoint>(Eigen::VectorXd::Random(3), ptSupportPoints);

  supportPoints[2] = std::make_shared<SupportPoint>(Eigen::VectorXd::Constant(3, 100.0)+Eigen::VectorXd::Random(3), ptSupportPoints);
  supportPoints[3] = std::make_shared<SupportPoint>(Eigen::VectorXd::Constant(3, 100.0)+Eigen::VectorXd::Random(3), ptSupportPoints);

  // try to create a support point cloud
  try {
    pt::ptree ptSupportPointCloud;
    ptSupportPointCloud.put("RequireConnectedGraphs", true);
    SupportPointCloud cloud(supportPoints, ptSupportPointCloud);
  } catch( exceptions::SupportPointCloudNotConnected const& exc ) {
    EXPECT_EQ(exc.outnum, 0);
  }
}
