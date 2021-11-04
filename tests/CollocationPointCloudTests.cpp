#include <gtest/gtest.h>

#include <stdio.h>

#include <MUQ/Utilities/HDF5/HDF5File.h>

#include <MUQ/Modeling/Distributions/RandomVariable.h>
#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/CollocationPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Utilities;
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

    // options for the collocation point cloud
    pt::ptree options;
    options.put("NumCollocationPoints", nCollocPoints);

    // the distribution we sample the colocation points from
    auto sampler = std::make_shared<CollocationPointSampler>(dist, model);

    // the cloud of collocation points
    collocationCloud = std::make_shared<CollocationPointCloud>(sampler, supportCloud, options);
  }

  virtual void TearDown() override {
    EXPECT_EQ(collocationCloud->NumPoints(), nCollocPoints);

    // loop through the support ponts
    std::size_t totalCollocPoints = 0;
    for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {
      auto suppi = supportCloud->GetSupportPoint(i);

      EXPECT_TRUE(collocationCloud->NumCollocationPerSupport(i)<=collocationCloud->NumPoints());
      totalCollocPoints += collocationCloud->NumCollocationPerSupport(i);

      std::vector<std::shared_ptr<CollocationPoint> > collocPoints = collocationCloud->CollocationPerSupport(i);
      EXPECT_EQ(collocPoints.size(), collocationCloud->NumCollocationPerSupport(i));
      for( const auto& pnt : collocPoints ) {
        EXPECT_TRUE(pnt);
        for( std::size_t j=0; j<supportCloud->NumPoints(); ++j ) {
          auto suppj = supportCloud->GetSupportPoint(j);
          EXPECT_TRUE((pnt->x-suppi->x).norm()<=(pnt->x-suppj->x).norm()+1.0e-14);
        }
      }
    }
    EXPECT_EQ(totalCollocPoints, collocationCloud->NumPoints());

    // loop through the collocation ponts
    for( auto it=collocationCloud->Begin(); it!=collocationCloud->End(); ++it ) {
      EXPECT_TRUE(*it);
      EXPECT_EQ((*it)->model, model);

      auto point = std::dynamic_pointer_cast<CollocationPoint>(*it);
      EXPECT_TRUE(point);

      auto nearest = point->supportPoint.lock();
      EXPECT_TRUE(nearest);
    }
  }

  /// The domain dimension
  const std::size_t indim = 6;

  /// The output dimension
  const std::size_t outdim = 4;

  /// The number of collocation points
  const std::size_t nCollocPoints = 75;

  // The model (default to just using the identity)
  std::shared_ptr<Model> model;

  /// The support point cloud
  std::shared_ptr<SupportPointCloud> supportCloud;

  /// The collocation point cloud
  std::shared_ptr<CollocationPointCloud> collocationCloud;
};

TEST_F(CollocationPointCloudTests, GenerateCollocationPoints) {
  // just run the default test
}

TEST_F(CollocationPointCloudTests, WriteToFile) {
  std::string file = std::tmpnam(nullptr);
  file += ".h5";

  collocationCloud->WriteToFile(file);

  HDF5File hdf5(file);
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {
    if( !hdf5.IsDataSet("/collocation points/support point "+std::to_string(i)) ) { continue; }
    const Eigen::MatrixXd pnts = hdf5.ReadMatrix("/collocation points/support point "+std::to_string(i));

    for( std::size_t j=0; j<pnts.rows(); ++j ) { EXPECT_NEAR((collocationCloud->GetCollocationPoint(collocationCloud->GlobalIndex(j, i))->x - pnts.row(j).transpose()).norm(), 0.0, 1.0e-12); }
  }
  hdf5.Close();

  // remove the file after the test is done
  std::remove(file.c_str());
}

} // namespace tests
} // namespace clf
