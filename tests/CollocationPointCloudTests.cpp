#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/RandomVariable.h>
#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/CollocationPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;

namespace clf {
namespace tests {

  /// A class to run the tests for clf::CollocationPointCloud
class CollocationPointCloudTests : public::testing::Test {
protected:

  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<LinearModel>(modelOptions);

    pt::ptree ptSupportPoints;
    ptSupportPoints.put("BasisFunctions", "Basis1, Basis2, Basis3, Basis4");
    ptSupportPoints.put("Basis1.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis2.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis3.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis4.Type", "TotalOrderPolynomials");

    // the number of support points
    const std::size_t n = 50;

    // create a bunch of random support points
    std::vector<std::shared_ptr<SupportPoint> > supportPoints(n);
    auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
    for( std::size_t i=0; i<n; ++i ) { supportPoints[i] = SupportPoint::Construct(dist->Sample(), model, ptSupportPoints); }

    // create the support point cloud
    pt::ptree ptSupportPointCloud;
    supportCloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  virtual void TearDown() override {
  }

  /// The domain dimension
  const std::size_t indim = 6;

  /// The output dimension
  const std::size_t outdim = 4;

  // The model (default to just using the identity)
  std::shared_ptr<Model> model;

  /// The support point cloud
  std::shared_ptr<SupportPointCloud> supportCloud;
};

TEST_F(CollocationPointCloudTests, GenerateCollocationPoints) {
  // the number of collocation points
  const std::size_t nCollocPoints = 75;

  // options for the collocation point cloud
  pt::ptree options;
  options.put("NumCollocationPoints", nCollocPoints);

  // the distribution we sample the colocation points from
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  auto sampler = std::make_shared<CollocationPointSampler>(dist, model);

  // the cloud of collocation points
  auto collocationCloud = std::make_shared<CollocationPointCloud>(sampler, supportCloud, options);
  EXPECT_EQ(collocationCloud->numCollocationPoints, nCollocPoints);

  // sample the collocation points
  collocationCloud->Resample();

  // loop through the colocation ponts
  for( auto it=collocationCloud->Begin(); it!=collocationCloud->End(); ++it ) {
    EXPECT_TRUE(*it);
    EXPECT_EQ((*it)->model, model);

    auto point = std::dynamic_pointer_cast<CollocationPoint>(*it);
    EXPECT_TRUE(point);

    auto nearest = point->supportPoint.lock();
    EXPECT_TRUE(nearest);
  }
}

} // namespace tests 
} // namespace clf
