#include <gtest/gtest.h>

#include "clf/UtilityFunctions.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"
#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class SupportPointTests : public::testing::Test {
protected:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt.put("OutputDimension", outdim);

    // choose a random location
    x = Eigen::VectorXd::Random(indim);
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {
    EXPECT_NEAR((point->x-x).norm(), 0.0, 1.0e-12);
    EXPECT_EQ(point->InputDimension(), indim);
    EXPECT_EQ(point->OutputDimension(), outdim);
  }

  /// The input dimension
  const std::size_t indim = 4;

  /// The output dimension
  const std::size_t outdim = 3;

  /// Options for the support point
  pt::ptree pt;

  /// The location of the support point
  Eigen::VectorXd x;

  /// The support point
  std::shared_ptr<SupportPoint> point;
};

TEST_F(SupportPointTests, LocalCoordinateTransformation) {
  // create the support point
  pt.put("BasisFunctions.Type", "TotalOrderPolynomials");
  point = std::make_shared<SupportPoint>(x, pt);

  // the default delta is 1.0
  EXPECT_DOUBLE_EQ(point->Radius(), 1.0);

  // reset delta
  const double newdelta = 0.5;
  point->Radius() = newdelta;
  EXPECT_DOUBLE_EQ(point->Radius(), newdelta);

  // chose a nearby point and compute the local coordinate
  const Eigen::VectorXd y = point->x + 0.1*newdelta*Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd yhat = point->LocalCoordinate(y);
  EXPECT_NEAR(((y-point->x)/newdelta - yhat).norm(), 0.0, 1.0e-10);
  EXPECT_NEAR((point->GlobalCoordinate(yhat) - y).norm(), 0.0, 1.0e-10);
}

TEST_F(SupportPointTests, TotalOrderPolynomials) {
  // the order of the total order polynomial basis
  const std::size_t order = 3;

  // create the support point
  pt.put("BasisFunctions.Type", "TotalOrderPolynomials");
  pt.put("BasisFunctions.Order", order);
  point = std::make_shared<SupportPoint>(x, pt);
  EXPECT_TRUE(point->basis);
  EXPECT_TRUE(std::dynamic_pointer_cast<PolynomialBasis>(point->basis));
  // the formula for the expected number of basis functions: [dimension]+[order] choose [dimension], in this case the answer is 35 so we hard code it
  EXPECT_EQ(point->basis->NumBasisFunctions(), 35);
}

TEST_F(SupportPointTests, TotalOrderSineCosine) {
  // the order of the total order sine/cosine basis
  const std::size_t order = 2;

  // create the support point
  pt.put("BasisFunctions.Type", "TotalOrderSinCos");
  pt.put("BasisFunctions.Order", order);
  point = std::make_shared<SupportPoint>(x, pt);
  EXPECT_TRUE(point->basis);
  EXPECT_TRUE(std::dynamic_pointer_cast<SinCosBasis>(point->basis));
  // the formula for the expected number of basis functions: [dimension]+[2*order] choose [dimension], in this case the answer is 70 so we hard code it
  EXPECT_EQ(point->basis->NumBasisFunctions(), 70);
}

TEST(SupportPointExceptionHandlingTests, InvalidBasisCheck) {
  // choose a random location
  const Eigen::VectorXd x = Eigen::VectorXd::Random(9);

  const std::string basisName = "ReallyLongUniqueInvalidBasisNameThatShouldOnlyBeUsedForTesting";

  // options for the support point
  pt::ptree pt;
  pt.put("BasisFunctions.Type", basisName);

  // create the support point
  try {
    auto point = std::make_shared<SupportPoint>(x, pt);
  } catch( SupportPointBasisException const& exc ) {
    EXPECT_EQ(exc.basisType, UtilityFunctions::ToUpper(basisName));
  }
}
