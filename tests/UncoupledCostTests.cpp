#include <gtest/gtest.h>

#include "clf/UncoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"
#include "clf/LevenbergMarquardt.hpp"
#include "clf/NLoptOptimizer.hpp"

#include "TestModels.hpp"

namespace pt = boost::property_tree;

namespace clf { 
namespace tests {

/// A class that runs the tests for clf::UncoupledCost
class UncoupledCostTests : public::testing::Test {
public:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<tests::TwoDimensionalAlgebraicModel>(modelOptions);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 4, orderSinCos = 2;

    // must be odd so that the center is not a point on the grid
    const std::size_t npoints = 7;

    // options for the support point
    pt::ptree suppOptions;
    suppOptions.put("NumNeighbors", npoints*npoints+1);
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis1.LocalBasis", false);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);
    suppOptions.put("RegularizationParameter", 0.0);
    suppOptions.put("UncoupledScale", uncoupledScale);
    point = SupportPoint::Construct(
      Eigen::VectorXd::Ones(indim),
      std::make_shared<tests::TwoDimensionalAlgebraicModel>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(suppOptions.get<std::size_t>("NumNeighbors"));
    supportPoints[0] = point;
    // add points on a grid so we know that they are well-poised
    for( std::size_t i=0; i<npoints; ++i ) {
      for( std::size_t j=0; j<npoints; ++j ) {
        supportPoints[i*npoints+j+1] = SupportPoint::Construct(
          point->x+0.1*Eigen::Vector2d((double)i/npoints-0.5, (double)j/npoints-0.5),
          model,
          suppOptions);
      }
    }
    pt::ptree ptSupportPointCloud;
    cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  /// Check the cost function
  virtual void TearDown() override {
    EXPECT_EQ(cost->inputDimension, point->NumCoefficients());
    EXPECT_DOUBLE_EQ(cost->UncoupledScale(), uncoupledScale);

    // the points should be the same
    auto costPt = cost->point.lock();
    EXPECT_NEAR((costPt->x-point->x).norm(), 0.0, 1.0e-10);
  }

  /// The input dimension
  const std::size_t indim = 2;

  /// The output dimension
  const std::size_t outdim = 2;

  /// The (nonlinear) model that we are using to test the uncoupled cost 
  std::shared_ptr<Model> model;

  /// A list of the support points (all of them are nearest neighborts to UncoupledCostTests::point)
  std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  /// The point that is associated with this cost function 
  std::shared_ptr<SupportPoint> point;
  
  /// A cloud that contains all of the support points
  std::shared_ptr<SupportPointCloud> cloud;

  /// The scale parameter that multiplies the residual in the cost function
  const double uncoupledScale = 1.25;

  /// The cost function that we are testing 
  std::shared_ptr<UncoupledCost> cost;
};

TEST_F(UncoupledCostTests, CostEvaluationAndDerivatives_ZeroRegularization) {
  // create the uncoupled cost
  pt::ptree costOptions;
  costOptions.put("UncoupledScale", uncoupledScale);
  cost = std::make_shared<UncoupledCost>(point, costOptions);
  EXPECT_EQ(cost->numPenaltyFunctions, point->NumNeighbors());
  EXPECT_DOUBLE_EQ(cost->RegularizationScale(), 0.0);

  // choose the vector of coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(point->NumCoefficients());

  // compute the true cost
  Eigen::VectorXd trueCost(point->NumNeighbors());
  {
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t i=0; i<supportPoints.size(); ++i ) {
      trueCost(i) = std::sqrt(0.5*uncoupledScale*kernel(i))*(model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients, point->GetBasisFunctions()) - model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x)).norm();
    }
  }

  const Eigen::VectorXd computedCost = cost->CostVector(coefficients);
  EXPECT_EQ(computedCost.size(), point->NumNeighbors());
  EXPECT_EQ(computedCost.size(), trueCost.size());
  for( std::size_t i=0; i<computedCost.size(); ++i ) { EXPECT_NEAR(computedCost(i), trueCost(i), 1.0e-12); }
    
  for( std::size_t i=0; i<supportPoints.size(); ++i ) {
    const Eigen::VectorXd gradFD = cost->PenaltyFunctionGradientByFD(i, coefficients);
    const Eigen::VectorXd grad = cost->PenaltyFunctionGradient(i, coefficients);
    EXPECT_NEAR((grad-gradFD).norm(), 0.0, 1.0e-5);
  }
}


TEST_F(UncoupledCostTests, CostEvaluationAndDerivatives_NonZeroRegularization) {
  // the regularization parameter for the uncoupled cost 
  const double regularizationScale = 0.5;

  // create the uncoupled cost
  pt::ptree costOptions;
  costOptions.put("RegularizationParameter", regularizationScale);
  costOptions.put("UncoupledScale", uncoupledScale);
  cost = std::make_shared<UncoupledCost>(point, costOptions);
  EXPECT_EQ(cost->numPenaltyFunctions, point->NumNeighbors()+1);
  EXPECT_DOUBLE_EQ(cost->RegularizationScale(), regularizationScale);

  // choose the vector of coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(point->NumCoefficients());

  // compute the true cost
  Eigen::VectorXd trueCost(point->NumNeighbors()+1);
  {
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t i=0; i<supportPoints.size(); ++i ) {
      trueCost(i) = std::sqrt(0.5*uncoupledScale*kernel(i))*(model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients, point->GetBasisFunctions()) - model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x)).norm();
    }

    trueCost(supportPoints.size()) = std::sqrt(0.5*regularizationScale)*coefficients.norm();
  }

  const Eigen::VectorXd computedCost = cost->CostVector(coefficients);
  EXPECT_EQ(computedCost.size(), point->NumNeighbors()+1);
  EXPECT_EQ(computedCost.size(), trueCost.size());
  for( std::size_t i=0; i<computedCost.size(); ++i ) { EXPECT_NEAR(computedCost(i), trueCost(i), 1.0e-12); }

  for( std::size_t i=0; i<supportPoints.size(); ++i ) {
    const Eigen::VectorXd gradFD = cost->PenaltyFunctionGradientByFD(i, coefficients);
    const Eigen::VectorXd grad = cost->PenaltyFunctionGradient(i, coefficients);
    EXPECT_NEAR((grad-gradFD).norm(), 0.0, 1.0e-5);
  }

  const Eigen::VectorXd gradFD = cost->PenaltyFunctionGradientByFD(supportPoints.size(), coefficients);
  const Eigen::VectorXd grad = cost->PenaltyFunctionGradient(supportPoints.size(), coefficients);
  EXPECT_NEAR((grad-gradFD).norm(), 0.0, 1.0e-5);
}

TEST_F(UncoupledCostTests, MinimizeCost_LevenbergMarquardt) {
  pt::ptree costOptions;
  costOptions.put("UncoupledScale", uncoupledScale);
  cost = std::make_shared<UncoupledCost>(point, costOptions);

  pt::ptree pt;
  pt.put("FunctionTolerance", 1.0e-9);
  auto lm = std::make_shared<DenseLevenbergMarquardt>(cost, pt);

  // choose the vector of coefficients
  Eigen::VectorXd coefficients = Eigen::VectorXd::Random(point->NumCoefficients());

  const std::pair<Optimization::Convergence, double> info = lm->Minimize(coefficients);
  EXPECT_TRUE(info.first>0);
  EXPECT_NEAR(info.second, 0.0, 1.0e-8);

  // the model at the optimal coefficients should equal its right hand side
  const Eigen::VectorXd eval = model->Operator(point->x, coefficients, point->GetBasisFunctions());
  const Eigen::VectorXd rhs = model->RightHandSide(point->x);
  EXPECT_NEAR(uncoupledScale*(eval-rhs).dot(eval-rhs), 0.0, 1.0e-6);
}

TEST_F(UncoupledCostTests, MinimizeCost_NLopt) {
  pt::ptree costOptions;
  costOptions.put("UncoupledScale", uncoupledScale);
  cost = std::make_shared<UncoupledCost>(point, costOptions);

  pt::ptree pt;
  pt.put("FunctionTolerance", 1.0e-12);
  pt.put("GradientTolerance", 1.0e-8);
  //pt.put("Algorithm", "NM");
  auto lm = std::make_shared<DenseNLoptOptimizer>(cost, pt);

  // choose the vector of coefficients
  Eigen::VectorXd coefficients = Eigen::VectorXd::Random(point->NumCoefficients());

  const std::pair<Optimization::Convergence, double> info = lm->Minimize(coefficients);
  EXPECT_TRUE(info.first>0);
  EXPECT_NEAR(info.second, 0.0, 1.0e-8);

  // the model at the optimal coefficients should equal its right hand side
  const Eigen::VectorXd eval = model->Operator(point->x, coefficients, point->GetBasisFunctions());
  const Eigen::VectorXd rhs = model->RightHandSide(point->x);
  EXPECT_NEAR(uncoupledScale*(eval-rhs).dot(eval-rhs), 0.0, 1.0e-6);
}

} // namespace tests 
} // namespace clf
