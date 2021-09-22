#include <gtest/gtest.h>

#include "clf/SupportPointCloud.hpp"
#include "clf/CoupledCost.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class ExampleModelForCoupledCostTests : public Model {
public:

  inline ExampleModelForCoupledCostTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForCoupledCostTests() = default;

protected:

private:
};

class CoupledCostTests : public::testing::Test {
public:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 5, orderSinCos = 2;

    // options for the support point
    pt::ptree suppOptions;
    suppOptions.put("NumNeighbors", npoints*npoints+1);
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis1.LocalBasis", false);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);
    point = SupportPoint::Construct(
      Eigen::VectorXd::Random(indim),
      std::make_shared<ExampleModelForCoupledCostTests>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(4*npoints*npoints+1);
    supportPoints[0] = point;
    // add points on a grid so we know that they are well-poised---make sure there is an even number of points on each side so that the center point is not on the grid
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j+1] = SupportPoint::Construct(
          point->x+0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          std::make_shared<ExampleModelForCoupledCostTests>(modelOptions),
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

  const double couplingScale = 0.25;

  const std::size_t npoints = 6;

private:
};

TEST_F(CoupledCostTests, CostEvaluationAndDerivatives) {
  // create the uncoupled cost for each neighbor
  for( const auto& it : supportPoints ) {
    pt::ptree costOptions;
    costOptions.put("CoupledScale", couplingScale);
    auto cost = std::make_shared<CoupledCost>(point, it, costOptions);
    EXPECT_EQ(cost->Coupled(), it!=point & point->IsNeighbor(it->GlobalIndex()));
    EXPECT_EQ(cost->inputDimension, point->NumCoefficients()+it->NumCoefficients());
    EXPECT_EQ(cost->numPenaltyFunctions, outdim);
    EXPECT_EQ(cost->GetPoint(), point);
    EXPECT_EQ(cost->GetNeighbor(), it);

    // choose the vector of coefficients
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(point->NumCoefficients() + it->NumCoefficients());

    const Eigen::VectorXd computedCost = cost->CostVector(coefficients);
    EXPECT_EQ(computedCost.size(), cost->numPenaltyFunctions);
    if( !cost->Coupled() ) {
      EXPECT_NEAR(computedCost.norm(), 0.0, 1.0e-14);
    } else {
      const Eigen::VectorXd pntEval = point->EvaluateLocalFunction(it->x, coefficients.head(point->NumCoefficients()));
      const Eigen::VectorXd neighEval = it->EvaluateLocalFunction(it->x, coefficients.tail(it->NumCoefficients()));
      const Eigen::VectorXd diff = std::sqrt(couplingScale*point->NearestNeighborKernel(point->LocalIndex(it->GlobalIndex())))*(pntEval - neighEval);
      EXPECT_EQ(diff.size(), cost->numPenaltyFunctions);

      EXPECT_NEAR(computedCost(0), diff(0), 1.0e-14);
      EXPECT_NEAR(computedCost(1), diff(1), 1.0e-14);
    }

    Eigen::MatrixXd jacFD = Eigen::MatrixXd::Zero(point->model->outputDimension, point->NumCoefficients()+it->NumCoefficients());
    EXPECT_EQ(jacFD.rows(), cost->numPenaltyFunctions);
    EXPECT_EQ(jacFD.cols(), cost->inputDimension);
    if( cost->Coupled() ) {
      const double dc = 1.0e-6;
      const Eigen::VectorXd kernel = point->NearestNeighborKernel();
      for( std::size_t c=0; c<point->NumCoefficients()+it->NumCoefficients(); ++c ) {
        Eigen::VectorXd coeffsp = coefficients;
        coeffsp(c) += dc;
        Eigen::VectorXd coeffsm = coefficients;
        coeffsm(c) -= dc;
        Eigen::VectorXd coeffs2m = coefficients;
        coeffs2m(c) -= 2.0*dc;

        const Eigen::VectorXd pntEvalp = point->EvaluateLocalFunction(it->x, coeffsp.head(point->NumCoefficients()));
        const Eigen::VectorXd neighEvalp = it->EvaluateLocalFunction(it->x, coeffsp.tail(it->NumCoefficients()));
        const Eigen::VectorXd diffp = pntEvalp - neighEvalp;

        const Eigen::VectorXd pntEval = point->EvaluateLocalFunction(it->x, coefficients.head(point->NumCoefficients()));
        const Eigen::VectorXd neighEval = it->EvaluateLocalFunction(it->x, coefficients.tail(it->NumCoefficients()));
        const Eigen::VectorXd diff = pntEval - neighEval;

        const Eigen::VectorXd pntEvalm = point->EvaluateLocalFunction(it->x, coeffsm.head(point->NumCoefficients()));
        const Eigen::VectorXd neighEvalm = it->EvaluateLocalFunction(it->x, coeffsm.tail(it->NumCoefficients()));
        const Eigen::VectorXd diffm = pntEvalm - neighEvalm;

        const Eigen::VectorXd pntEval2m = point->EvaluateLocalFunction(it->x, coeffs2m.head(point->NumCoefficients()));
        const Eigen::VectorXd neighEval2m = it->EvaluateLocalFunction(it->x, coeffs2m.tail(it->NumCoefficients()));
        const Eigen::VectorXd diff2m = pntEval2m - neighEval2m;

        jacFD.col(c) = std::sqrt(couplingScale*point->NearestNeighborKernel(point->LocalIndex(it->GlobalIndex())))*(2.0*diffp + 3.0*diff - 6.0*diffm + diff2m)/(6.0*dc);
      }
    }

    Eigen::SparseMatrix<double> jac;
    cost->Jacobian(coefficients, jac);

    EXPECT_EQ(jac.rows(), cost->numPenaltyFunctions);
    EXPECT_EQ(jac.cols(), cost->inputDimension);
    for( std::size_t i=0; i<cost->numPenaltyFunctions; ++i ) {
      for( std::size_t j=0; j<cost->inputDimension; ++j ) {
        EXPECT_NEAR(jac.coeff(i, j), jacFD(i, j), 1.0e-8);
      }
    }
  }
}
