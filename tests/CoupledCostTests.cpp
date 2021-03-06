#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/SupportPointCloud.hpp"
#include "clf/CoupledCost.hpp"

namespace pt = boost::property_tree;

namespace clf {
namespace tests {

/// A class to run the tests for clf::CoupledCost
class CoupledCostTests : public::testing::Test {
public:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<LinearModel>(modelOptions);

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
    point = SupportPoint::Construct(Eigen::VectorXd::Random(indim), model, suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    std::vector<std::shared_ptr<SupportPoint> > supportPoints(4*npoints*npoints+1);
    supportPoints[0] = point;
    // add points on a grid---make sure there is an even number of points on each side so that the center point is not on the grid
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j+1] = SupportPoint::Construct(
          point->x+0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          model, suppOptions);
      }
    }
    pt::ptree ptSupportPointCloud;
    cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  /// The input dimension
  const std::size_t indim = 2;

  /// The output dimension
  const std::size_t outdim = 2;

  /// Associate the coupled cost with this point
  std::shared_ptr<SupportPoint> point;

  /// A support point cloud that holds all the support points
  std::shared_ptr<SupportPointCloud> cloud;

  /// The coupling scale (parameter \f$c_i\f$ in the coupled cost---see clf::CoupledCost)
  const double couplingScale = 0.25;

  /// The number of points in the cloud will be <tt>4*npoints*npoints+1</tt>
  const std::size_t npoints = 6;

  /// The model that we are using to test the coupling cost
  std::shared_ptr<LinearModel> model;

private:
};

TEST_F(CoupledCostTests, CostEvaluationAndDerivatives) {
  // create the uncoupled cost for each neighbor
  for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
    // couple the point-of-interest with this neighbor
    auto coupledPoint = cloud->GetSupportPoint(i);
    EXPECT_TRUE(coupledPoint);

    pt::ptree costOptions;
    costOptions.put("CoupledScale", couplingScale);
    auto cost = std::make_shared<CoupledCost>(point, coupledPoint, costOptions);
    EXPECT_TRUE(cost->IsQuadratic()); // we use a linear model in this test
    EXPECT_EQ(cost->Coupled(), coupledPoint!=point & point->IsNeighbor(coupledPoint->GlobalIndex()));
    EXPECT_EQ(cost->inputDimension, point->NumCoefficients()+coupledPoint->NumCoefficients());
    EXPECT_EQ(cost->numPenaltyFunctions, 1);
    EXPECT_EQ(cost->numPenaltyTerms, model->outputDimension);
    EXPECT_EQ(cost->GetPoint(), point);
    EXPECT_EQ(cost->GetNeighbor(), coupledPoint);
    EXPECT_NEAR(cost->CoupledScale(), (cost->Coupled()? couplingScale : 0.0), 1.0e-14);

    // choose the vector of coefficients
    const Eigen::VectorXd pointCoeffs = Eigen::VectorXd::Random(point->NumCoefficients());
    const Eigen::VectorXd neighCoeffs = Eigen::VectorXd::Random(coupledPoint->NumCoefficients());
    Eigen::VectorXd coefficients(point->NumCoefficients() + coupledPoint->NumCoefficients());
    coefficients.head(point->NumCoefficients()) = pointCoeffs;
    coefficients.tail(coupledPoint->NumCoefficients()) = neighCoeffs;

    // evaluate the penalty function
    const Eigen::VectorXd penaltyFunc0 = cost->CostFunction::PenaltyFunction(0, coefficients);
    const Eigen::VectorXd penaltyFunc1 = cost->PenaltyFunction(pointCoeffs, neighCoeffs);
    EXPECT_EQ(penaltyFunc0.size(), penaltyFunc1.size());
    EXPECT_NEAR((penaltyFunc0-penaltyFunc1).norm(), 0.0, 1.0e-14);

    const Eigen::VectorXd expectedPenaltyFunc = (cost->Coupled()?
    std::sqrt(0.5*couplingScale*point->NearestNeighborKernel(point->LocalIndex(coupledPoint->GlobalIndex())))*(point->EvaluateLocalFunction(coupledPoint->x, pointCoeffs) - coupledPoint->EvaluateLocalFunction(coupledPoint->x, neighCoeffs))
    :
    Eigen::VectorXd::Zero(coupledPoint->model->outputDimension).eval() );
    EXPECT_EQ(penaltyFunc0.size(), expectedPenaltyFunc.size());
    EXPECT_NEAR((penaltyFunc0-expectedPenaltyFunc).norm(), 0.0, 1.0e-14);

    // compute the gradient of the penalty function
    const Eigen::MatrixXd jacFD = cost->PenaltyFunctionJacobianByFD(0, coefficients);
    const Eigen::MatrixXd jac0 = cost->CostFunction::PenaltyFunctionJacobian(0, coefficients);
    EXPECT_EQ(jacFD.rows(), jac0.rows());
    EXPECT_EQ(jacFD.cols(), jac0.cols());
    const std::vector<Eigen::Triplet<double> > entries = cost->PenaltyFunctionJacobian();
    Eigen::SparseMatrix<double> jac1(jac0.rows(), jac0.cols());
    jac1.setFromTriplets(entries.begin(), entries.end());
    EXPECT_NEAR((jac0-Eigen::MatrixXd(jac1)).norm(), 0.0, 1.0e-14);
    EXPECT_NEAR((jac0-jacFD).norm(), 0.0, 1.0e-6);
  }
}

} // namespace tests
} // namespace clf
