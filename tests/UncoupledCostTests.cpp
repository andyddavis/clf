#include <gtest/gtest.h>

#include "clf/UncoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"

#include "TestModels.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class UncoupledCostTests : public::testing::Test {
public:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

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
          std::make_shared<tests::TwoDimensionalAlgebraicModel>(modelOptions),
          suppOptions);
      }
    }
    pt::ptree ptSupportPointCloud;
    cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {}

  /// The input and output dimensions
  const std::size_t indim = 2, outdim = 2;

  std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  std::shared_ptr<SupportPoint> point;

  std::shared_ptr<SupportPointCloud> cloud;

  const double regularizationScale = 0.5;

  const double uncoupledScale = 0.25;
};

TEST_F(UncoupledCostTests, CostEvaluationAndDerivatives_ZeroRegularization) {
  // create the uncoupled cost
  pt::ptree costOptions;
  costOptions.put("RegularizationParameter", 0.0);
  costOptions.put("UncoupledScale", uncoupledScale);
  auto cost = std::make_shared<UncoupledCost>(point, costOptions);
  EXPECT_EQ(cost->inputDimension, point->NumCoefficients());
  EXPECT_EQ(cost->numPenaltyFunctions, point->NumNeighbors()*point->model->outputDimension);
  EXPECT_DOUBLE_EQ(cost->RegularizationScale(), 0.0);
  EXPECT_DOUBLE_EQ(cost->UncoupledScale(), uncoupledScale);

  // the points should be the same
  auto costPt = cost->point.lock();
  EXPECT_NEAR((costPt->x-point->x).norm(), 0.0, 1.0e-10);

  // choose the vector of coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(point->NumCoefficients());

  // compute the true cost
  Eigen::VectorXd trueCost(point->NumNeighbors()*point->model->outputDimension);
  {
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t i=0; i<supportPoints.size(); ++i ) {
      trueCost.segment(i*point->model->outputDimension, point->model->outputDimension) = std::sqrt(uncoupledScale*kernel(i))*(supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients, point->GetBasisFunctions()) - supportPoints[point->GlobalNeighborIndex(i)]->model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x));
    }
  }

  const Eigen::VectorXd computedCost = cost->CostVector(coefficients);
  EXPECT_EQ(computedCost.size(), point->NumNeighbors()*point->model->outputDimension);
  EXPECT_EQ(computedCost.size(), trueCost.size());
  for( std::size_t i=0; i<computedCost.size(); ++i ) { EXPECT_NEAR(computedCost(i), trueCost(i), 1.0e-12); }

  Eigen::MatrixXd jacFD = Eigen::MatrixXd::Zero(point->NumNeighbors()*point->model->outputDimension, point->NumCoefficients());
  {
    const double dc = 1.0e-8;
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t c=0; c<point->NumCoefficients(); ++c ) {
      Eigen::VectorXd coeffsp = coefficients;
      coeffsp(c) += dc;
      Eigen::VectorXd coeffsm = coefficients;
      coeffsm(c) -= dc;
      Eigen::VectorXd coeffs2m = coefficients;
      coeffs2m(c) -= 2.0*dc;
      for( std::size_t i=0; i<supportPoints.size(); ++i ) {
        const Eigen::VectorXd rhs = supportPoints[point->GlobalNeighborIndex(i)]->model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x);
        const Eigen::VectorXd diffp = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coeffsp, point->GetBasisFunctions()) - rhs;
        const Eigen::VectorXd diff = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients, point->GetBasisFunctions()) - rhs;
        const Eigen::VectorXd diffm = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coeffsm, point->GetBasisFunctions()) - rhs;
        const Eigen::VectorXd diff2m = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coeffs2m, point->GetBasisFunctions()) - rhs;

        for( std::size_t j=0; j<point->model->outputDimension; ++j ) {
          jacFD(i*point->model->outputDimension+j, c) = std::sqrt(uncoupledScale*kernel(i))*(2.0*diffp(j) + 3.0*diff(j) - 6.0*diffm(j) + diff2m(j))/(6.0*dc);
        }
      }
    }
  }

  Eigen::SparseMatrix<double> jac;
  cost->Jacobian(coefficients, jac);
  EXPECT_EQ(jac.rows(), cost->numPenaltyFunctions);
  EXPECT_EQ(jac.rows(), jacFD.rows());
  EXPECT_EQ(jac.cols(), cost->inputDimension);
  EXPECT_EQ(jac.cols(), jacFD.cols());

  for( std::size_t i=0; i<cost->numPenaltyFunctions; ++i ) {
    for( std::size_t j=0; j<cost->inputDimension; ++j ) {
      EXPECT_NEAR(jac.coeff(i, j), jacFD(i, j), 1.0e-6);
    }
  }
}

TEST_F(UncoupledCostTests, CostEvaluationAndDerivatives_NonZeroRegularization) {
  // create the uncoupled cost
  pt::ptree costOptions;
  costOptions.put("RegularizationParameter", regularizationScale);
  costOptions.put("UncoupledScale", uncoupledScale);
  auto cost = std::make_shared<UncoupledCost>(point, costOptions);
  EXPECT_EQ(cost->inputDimension, point->NumCoefficients());
  EXPECT_EQ(cost->numPenaltyFunctions, point->NumNeighbors()*point->model->outputDimension+point->NumCoefficients());
  EXPECT_DOUBLE_EQ(cost->RegularizationScale(), regularizationScale);
  EXPECT_DOUBLE_EQ(cost->UncoupledScale(), uncoupledScale);

  // the points should be the same
  auto costPt = cost->point.lock();
  EXPECT_NEAR((costPt->x-point->x).norm(), 0.0, 1.0e-10);

  // choose the vector of coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(point->NumCoefficients());

  // compute the true cost
  Eigen::VectorXd trueCost(point->NumNeighbors()*point->model->outputDimension+point->NumCoefficients());
  {
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t i=0; i<supportPoints.size(); ++i ) {
      trueCost.segment(i*point->model->outputDimension, point->model->outputDimension) = std::sqrt(uncoupledScale*kernel(i))*(supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients, point->GetBasisFunctions()) - supportPoints[point->GlobalNeighborIndex(i)]->model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x));
    }
    trueCost.tail(point->NumCoefficients()) = std::sqrt(regularizationScale)*coefficients;
  }

  const Eigen::VectorXd computedCost = cost->CostVector(coefficients);
  EXPECT_EQ(computedCost.size(), point->NumNeighbors()*point->model->outputDimension+point->NumCoefficients());
  EXPECT_EQ(computedCost.size(), trueCost.size());
  for( std::size_t i=0; i<computedCost.size(); ++i ) { EXPECT_NEAR(computedCost(i), trueCost(i), 1.0e-12); }

  Eigen::MatrixXd jacFD = Eigen::MatrixXd::Zero(point->NumNeighbors()*point->model->outputDimension+point->NumCoefficients(), point->NumCoefficients());
  {
    const double dc = 1.0e-8;
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t c=0; c<point->NumCoefficients(); ++c ) {
      Eigen::VectorXd coeffsp = coefficients;
      coeffsp(c) += dc;
      Eigen::VectorXd coeffsm = coefficients;
      coeffsm(c) -= dc;
      Eigen::VectorXd coeffs2m = coefficients;
      coeffs2m(c) -= 2.0*dc;
      for( std::size_t i=0; i<supportPoints.size(); ++i ) {
        const Eigen::VectorXd rhs = supportPoints[point->GlobalNeighborIndex(i)]->model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x);
        const Eigen::VectorXd diffp = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coeffsp, point->GetBasisFunctions()) - rhs;
        const Eigen::VectorXd diff = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients, point->GetBasisFunctions()) - rhs;
        const Eigen::VectorXd diffm = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coeffsm, point->GetBasisFunctions()) - rhs;
        const Eigen::VectorXd diff2m = supportPoints[point->GlobalNeighborIndex(i)]->model->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coeffs2m, point->GetBasisFunctions()) - rhs;

        for( std::size_t j=0; j<point->model->outputDimension; ++j ) {
          jacFD(i*point->model->outputDimension+j, c) = std::sqrt(uncoupledScale*kernel(i))*(2.0*diffp(j) + 3.0*diff(j) - 6.0*diffm(j) + diff2m(j))/(6.0*dc);
        }
      }
    }

    jacFD.block(point->NumNeighbors()*point->model->outputDimension, 0, point->NumCoefficients(), point->NumCoefficients()) = std::sqrt(regularizationScale)*Eigen::MatrixXd::Identity(point->NumCoefficients(), point->NumCoefficients());
  }
  EXPECT_EQ(jacFD.rows(), cost->numPenaltyFunctions);
  EXPECT_EQ(jacFD.cols(), cost->inputDimension);

  Eigen::SparseMatrix<double> jac;
  cost->Jacobian(coefficients, jac);
  EXPECT_EQ(jac.rows(), cost->numPenaltyFunctions);
  EXPECT_EQ(jac.cols(), cost->inputDimension);

  for( std::size_t i=0; i<cost->numPenaltyFunctions; ++i ) {
    for( std::size_t j=0; j<cost->inputDimension; ++j ) {
      EXPECT_NEAR(jac.coeff(i, j), jacFD(i, j), 1.0e-4);
    }
  }
}

TEST_F(UncoupledCostTests, MinimizeOnePoint) {
  assert(point);
  const double cost = point->MinimizeUncoupledCost();
  EXPECT_NEAR(cost, 0.0, 1.0e-8);

  for( const auto& it : supportPoints ) {
    const Eigen::VectorXd eval = point->EvaluateLocalFunction(it->x);
    const Eigen::VectorXd operatorEval = Eigen::Vector2d(eval(0), eval(1)+eval(1)*eval(1));
    const Eigen::VectorXd expectedRHS = Eigen::Vector2d(
      std::sin(2.0*M_PI*it->x(0))*std::cos(M_PI*it->x(1)) + std::cos(it->x(0)),
      it->x.prod()
    );
    EXPECT_EQ(eval.size(), point->model->outputDimension);
    for( std::size_t i=0; i<eval.size(); ++i ) { EXPECT_NEAR(expectedRHS(i), operatorEval(i), std::sqrt(cost)); }
  }
}
