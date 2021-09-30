#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/RandomVariable.h>
#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/ColocationPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

class ColocationPointCloudTests : public::testing::Test {
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

TEST_F(ColocationPointCloudTests, GenerateColocationPoints) {
  // the number of colocation points
  const std::size_t nColocPoints = 75;

  // options for the colocation point cloud
  pt::ptree options;
  options.put("NumColocationPoints", nColocPoints);

  // the distribution we sample the colocation points from
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  auto sampler = std::make_shared<ColocationPointSampler>(dist, model);

  // the cloud of collocation points
  auto colocationCloud = std::make_shared<ColocationPointCloud>(sampler, supportCloud, options);
  EXPECT_EQ(colocationCloud->numColocationPoints, nColocPoints);
  EXPECT_EQ(colocationCloud->InputDimension(), model->inputDimension);
  EXPECT_EQ(colocationCloud->OutputDimension(), model->outputDimension);

  // sample the colocation points
  colocationCloud->Resample();

  // loop through the colocation ponts
  for( auto it=colocationCloud->Begin(); it!=colocationCloud->End(); ++it ) {
    EXPECT_TRUE(*it);
    EXPECT_EQ((*it)->model, model);

    auto nearest = (*it)->supportPoint.lock();
    EXPECT_TRUE(nearest);
  }
}
