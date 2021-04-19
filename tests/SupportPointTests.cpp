#include <gtest/gtest.h>

#include "clf/UtilityFunctions.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"
#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class ExampleModelForSupportPointTests : public Model {
public:

  inline ExampleModelForSupportPointTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForSupportPointTests() = default;
private:
};

class SupportPointTests : public::testing::Test {
protected:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<ExampleModelForSupportPointTests>(modelOptions);

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
  const std::size_t outdim = 3;

  /// Options for the support point
  pt::ptree pt;

  /// The location of the support point
  Eigen::VectorXd x;

  /// The model for the support point
  std::shared_ptr<Model> model;

  /// The support point
  std::shared_ptr<SupportPoint> point;
};

TEST_F(SupportPointTests, LocalCoordinateTransformation) {
  // create the support point
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderPolynomials");
  pt.put("Basis2.Type", "TotalOrderPolynomials");
  pt.put("Basis3.Type", "TotalOrderPolynomials");
  point = std::make_shared<SupportPoint>(x, model, pt);

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

  // create the support points
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderPolynomials");
  pt.put("Basis1.Order", order);
  pt.put("Basis2.Type", "TotalOrderPolynomials");
  pt.put("Basis2.Order", order);
  pt.put("Basis3.Type", "TotalOrderPolynomials");
  pt.put("Basis3.Order", order);
  point = std::make_shared<SupportPoint>(x, model, pt);
  EXPECT_EQ(point->bases.size(), outdim);
  EXPECT_EQ(point->numNeighbors.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) {
    EXPECT_TRUE(point->bases[i]);
    EXPECT_TRUE(std::dynamic_pointer_cast<const PolynomialBasis>(point->bases[i]));
    // the formula for the expected number of basis functions: [dimension]+[order] choose [dimension], in this case the answer is 35 so we hard code it
    EXPECT_EQ(point->bases[i]->NumBasisFunctions(), 35);
    EXPECT_EQ(point->numNeighbors[i], 36);
  }
}

TEST_F(SupportPointTests, TotalOrderSineCosine) {
  // the order of the total order sin/cos basis
  const std::size_t order = 2;

  // create the support points
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderSinCos");
  pt.put("Basis1.Order", order);
  pt.put("Basis2.Type", "TotalOrderSinCos");
  pt.put("Basis2.Order", order);
  pt.put("Basis3.Type", "TotalOrderSinCos");
  pt.put("Basis3.Order", order);
  point = std::make_shared<SupportPoint>(x, model, pt);
  EXPECT_EQ(point->bases.size(), outdim);
  EXPECT_EQ(point->numNeighbors.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) {
    EXPECT_TRUE(point->bases[i]);
    EXPECT_TRUE(std::dynamic_pointer_cast<const SinCosBasis>(point->bases[i]));
    // the formula for the expected number of basis functions: [dimension]+[2*order] choose [dimension], in this case the answer is 70 so we hard code it
    EXPECT_EQ(point->bases[i]->NumBasisFunctions(), 70);
    EXPECT_EQ(point->numNeighbors[i], 71);
  }
}

TEST_F(SupportPointTests, MixedBasisTypes) {
  // the order of the total order polynomial and sin/cos bases
  const std::size_t orderPoly = 3, orderSinCos = 2;

  // create the support point// create the support point
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderSinCos");
  pt.put("Basis1.Order", orderSinCos);
  pt.put("Basis2.Type", "TotalOrderPolynomials");
  pt.put("Basis2.Order", orderPoly);
  pt.put("Basis3.Type", "TotalOrderSinCos");
  pt.put("Basis3.Order", orderSinCos);
  point = std::make_shared<SupportPoint>(x, model, pt);
  EXPECT_EQ(point->bases.size(), outdim);
  EXPECT_EQ(point->numNeighbors.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) {
    EXPECT_TRUE(point->bases[i]);
    if( i==1 ) {
      EXPECT_TRUE(std::dynamic_pointer_cast<const PolynomialBasis>(point->bases[i]));
      // the formula for the expected number of basis functions: [dimension]+[order] choose [dimension], in this case the answer is 35 so we hard code it
      EXPECT_EQ(point->bases[i]->NumBasisFunctions(), 35);
      EXPECT_EQ(point->numNeighbors[i], 36);
    } else {
      EXPECT_TRUE(std::dynamic_pointer_cast<const SinCosBasis>(point->bases[i]));
      // the formula for the expected number of basis functions: [dimension]+[2*order] choose [dimension], in this case the answer is 70 so we hard code it
      EXPECT_EQ(point->bases[i]->NumBasisFunctions(), 70);
      EXPECT_EQ(point->numNeighbors[i], 71);
    }
  }
}

TEST_F(SupportPointTests, CustomNearestNeighborsWithDefault) {
  // the order of the total order polynomial and sin/cos bases
  const std::size_t orderPoly = 3, orderSinCos = 2;

  // create the support point// create the support point
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderSinCos");
  pt.put("Basis2.Type", "TotalOrderPolynomials");
  pt.put("Basis3.Type", "TotalOrderSinCos");
  pt.put("NumNeighbors", 85);
  pt.put("NumNeighbors-2", 73);
  point = std::make_shared<SupportPoint>(x, model, pt);
  EXPECT_EQ(point->numNeighbors.size(), outdim);
  EXPECT_EQ(point->numNeighbors[0], 85);
  EXPECT_EQ(point->numNeighbors[1], 85);
  EXPECT_EQ(point->numNeighbors[2], 73);
}

TEST_F(SupportPointTests, CustomNearestNeighbors) {
  // the order of the total order polynomial and sin/cos bases
  const std::size_t orderPoly = 3, orderSinCos = 2;

  // create the support point// create the support point
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderSinCos");
  pt.put("Basis2.Type", "TotalOrderPolynomials");
  pt.put("Basis3.Type", "TotalOrderSinCos");
  pt.put("NumNeighbors-0", 85);
  pt.put("NumNeighbors-1", 45);
  pt.put("NumNeighbors-2", 73);
  point = std::make_shared<SupportPoint>(x, model, pt);
  EXPECT_EQ(point->numNeighbors.size(), outdim);
  EXPECT_EQ(point->numNeighbors[0], 85);
  EXPECT_EQ(point->numNeighbors[1], 45);
  EXPECT_EQ(point->numNeighbors[2], 73);
}

TEST(SupportPointExceptionHandlingTests, WrongNumberOfNearestNeighbors) {
  // the input and output dimensions
  const std::size_t indim = 4, outdim = 1;

  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<ExampleModelForSupportPointTests>(modelOptions);

  // options for the support point
  pt::ptree pt;
  pt.put("OutputDimension", outdim);
  pt.put("BasisFunctions", "Basis");
  pt.put("Basis.Type", "TotalOrderSinCos");
  pt.put("NumNeighbors", 1);

  // choose a random location
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // not enough bases
  try {
    auto point = std::make_shared<SupportPoint>(x, model, pt);
  } catch( exceptions::SupportPointWrongNumberOfNearestNeighbors const& exc ) {
    EXPECT_EQ(exc.output, 0);
    EXPECT_EQ(exc.required, 15);
    EXPECT_EQ(exc.supplied, 1);
  }
}

TEST(SupportPointExceptionHandlingTests, WrongNumberOfBases) {
  // the input and output dimensions
  const std::size_t indim = 4, outdim = 3;

  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<ExampleModelForSupportPointTests>(modelOptions);

  // options for the support point
  pt::ptree pt;
  pt.put("OutputDimension", outdim);

  // choose a random location
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // not enough bases
  try {
    pt.put("BasisFunctions", "Basis1, Basis2");
    auto point = std::make_shared<SupportPoint>(x, model, pt);
  } catch( exceptions::SupportPointWrongNumberOfBasesConstructed const& exc ) {
    EXPECT_EQ(exc.outdim, outdim);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.basisOptionNames, "Basis1,Basis2");
  }

  // too many bases
  try {
    pt.put("BasisFunctions", "Basis1, Basis2, Basis3, Basis4");
    auto point = std::make_shared<SupportPoint>(x, model, pt);
  } catch( exceptions::SupportPointWrongNumberOfBasesConstructed const& exc ) {
    EXPECT_EQ(exc.outdim, outdim);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.basisOptionNames, "Basis1,Basis2,Basis3,Basis4");
  }

  // leading comma
  try {
    pt.put("BasisFunctions", " , Basis1, Basis2");
    auto point = std::make_shared<SupportPoint>(x, model, pt);
  } catch( exceptions::SupportPointWrongNumberOfBasesConstructed const& exc ) {
    EXPECT_EQ(exc.outdim, outdim);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.basisOptionNames, "Basis1,Basis2");
  }

  // trailing comma
  try {
    pt.put("BasisFunctions", "Basis1, Basis2, ");
    auto point = std::make_shared<SupportPoint>(x, model, pt);
  } catch( exceptions::SupportPointWrongNumberOfBasesConstructed const& exc ) {
    EXPECT_EQ(exc.outdim, outdim);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.basisOptionNames, "Basis1,Basis2");
  }
}

TEST(SupportPointExceptionHandlingTests, InvalidBasisCheck) {
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", 1);
  modelOptions.put("OutputDimension", 1);
  auto model = std::make_shared<ExampleModelForSupportPointTests>(modelOptions);

  // choose a random location
  const Eigen::VectorXd x = Eigen::VectorXd::Random(9);

  const std::string basisName = "ReallyLongUniqueInvalidBasisNameThatShouldOnlyBeUsedForTesting";

  // options for the support point
  pt::ptree pt;
  pt.put("BasisFunctions", "Basis");
  pt.put("Basis.Type", basisName);

  // create the support point
  try {
    auto point = std::make_shared<SupportPoint>(x, model, pt);
  } catch( exceptions::SupportPointInvalidBasisException const& exc ) {
    EXPECT_EQ(exc.basisType, UtilityFunctions::ToUpper(basisName));
  }
}
