#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/CoupledSupportPoint.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

class CoupledSupportPointTests : public::testing::Test {
protected:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<LinearModel>(modelOptions);

    // choose a random location
    x = Eigen::VectorXd::Random(indim);
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {
    EXPECT_NEAR((point->x-x).norm(), 0.0, 1.0e-12);
    EXPECT_EQ(point->model->inputDimension, indim);
    EXPECT_EQ(point->model->outputDimension, outdim);
  }

  /// The input dimension
  const std::size_t indim = 4;

  /// The output dimension
  const std::size_t outdim = 1;

  /// Options for the support point
  pt::ptree pt;

  /// The location of the support point
  Eigen::VectorXd x;

  /// The model for the support point
  std::shared_ptr<Model> model;

  /// The support point
  std::shared_ptr<CoupledSupportPoint> point;

  /// The magnitude scaling parameter for the coupling function
  const double c0 = 0.2346;

  /// The exponential scaling parameter for the coupling function
  const double c1 = 0.94025;
};

TEST_F(CoupledSupportPointTests, CouplingFunction) {
  // create the support points
  pt.put("MagnitudeScale", c0);
  pt.put("ExponentialScale", c1);
  pt.put("BasisFunctions", "Basis");
  pt.put("Basis.Type", "TotalOrderPolynomials");
  pt.put("Basis.Order", 1);

  std::vector<std::shared_ptr<SupportPoint> > points(10);
  point = CoupledSupportPoint::Construct(x, model, pt);
  points.at(0) = point;

  auto rv = std::make_shared<Gaussian>(x, Eigen::MatrixXd::Identity(indim, indim));
  for( auto it=points.begin()+1; it!=points.end(); ++it ) { *it = CoupledSupportPoint::Construct(rv->Sample(), model, pt); }

  // before the cloud is constructed, the coupling function is zero
  for( const auto& it : points ) {
    for( std::size_t i=0; i<points.size(); ++i ) { EXPECT_NEAR(it->CouplingFunction(i), 0.0, 1.0e-12); }
  }

  pt::ptree cloudOptions;
  auto cloud = SupportPointCloud::Construct(points, cloudOptions);

  // after the cloud is constructed, the coupling function is nonzero for the neighbors
  for( const auto& it : points ) {
    for( std::size_t i=0; i<points.size(); ++i ) {
      const double maxDist = it->SquaredDistanceToNeighbor(it->NumNeighbors()-1);
      if( i>=it->NumNeighbors() ) {
        EXPECT_NEAR(it->CouplingFunction(i), 0.0, 1.0e-12);
      } else if( i==0 ) {
        EXPECT_NEAR(it->CouplingFunction(i), c0, 1.0e-12);
      } else {
        EXPECT_NEAR(it->CouplingFunction(i), c0*std::exp(c1*(1.0-1.0/(1.0-it->SquaredDistanceToNeighbor(i)/maxDist))), 1.0e-12);
      }
    }
  }
}
