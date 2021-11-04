#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/RandomVariable.h>
#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/CollocationCost.hpp"
#include "clf/LevenbergMarquardt.hpp"

#include "TestModels.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;

namespace clf {
namespace tests {

/// A class that runs the tests for clf::CollocationCost
class CollocationCostTests : public::testing::Test {
protected:
  virtual void SetUp() override {
    pt::ptree ptSupportPoints;
    ptSupportPoints.put("BasisFunctions", "Basis1, Basis2");
    ptSupportPoints.put("Basis1.Type", "TotalOrderSinCos");
    ptSupportPoints.put("Basis1.Order", 6);
    ptSupportPoints.put("Basis2.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis2.Order", 4);
    ptSupportPoints.put("OutputDimension", outdim);

    // the number of support points
    const std::size_t n = 100;

    // create a bunch of random support points
    std::vector<std::shared_ptr<SupportPoint> > supportPoints(n);
    auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
    for( std::size_t i=0; i<n; ++i ) { supportPoints[i] = SupportPoint::Construct(dist->Sample(), ptSupportPoints); }

    // create the support point cloud
    pt::ptree ptSupportPointCloud;
    supportCloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);

    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<tests::TwoDimensionalAlgebraicModel>(modelOptions);

    // the distribution we sample the colocation points from
    auto sampler = std::make_shared<CollocationPointSampler>(dist, model);

    // the number of collocation points
    const std::size_t nCollocPoints = 1250;
    
    // options for the cost function
    cloudOptions.put("NumCollocationPoints", nCollocPoints);
    collocationCloud = std::make_shared<CollocationPointCloud>(sampler, supportCloud, cloudOptions);

  }

  virtual void TearDown() override {
    //EXPECT_EQ(cost->inputDimension, model->outputDimension*supportCloud->NumPoints());
  }

  /// Options for the colocation point cloud function
  pt::ptree cloudOptions;

  /// The domain dimension
  const std::size_t indim = 2;

  /// The output dimension
  const std::size_t outdim = 2;

  // The model (default to just using the identity)
  std::shared_ptr<Model> model;

  /// The support point cloud
  std::shared_ptr<SupportPointCloud> supportCloud;

  /// The collocation point cloud
  std::shared_ptr<CollocationPointCloud> collocationCloud;
};

TEST_F(CollocationCostTests, ConstructAndEvaluate) {
  // create the collocation costs
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {
    auto support = supportCloud->GetSupportPoint(i);

    auto cost = std::make_shared<CollocationCost>(support, collocationCloud->CollocationPerSupport(i));
    EXPECT_EQ(cost->inputDimension, support->NumCoefficients());
    EXPECT_EQ(cost->numPenaltyFunctions, collocationCloud->NumCollocationPerSupport(i));
    EXPECT_EQ(cost->numPenaltyTerms, cost->numPenaltyFunctions*model->outputDimension);
    if( cost->numPenaltyFunctions>0 ) { EXPECT_FALSE(cost->IsQuadratic()); } else { EXPECT_TRUE(cost->IsQuadratic()); }

    // choose the vector of coefficients
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(support->NumCoefficients());

    const Eigen::VectorXd computedCost = cost->CostVector(coefficients);

    // compute the true cost
    Eigen::VectorXd trueCost(cost->numPenaltyTerms);
    {
      std::size_t cnt = 0;
      for( std::size_t j=0; j<collocationCloud->NumCollocationPerSupport(i); ++j ) {
	auto colloc = collocationCloud->GetCollocationPoint(collocationCloud->GlobalIndex(j, support->GlobalIndex()));

	trueCost.segment(cnt, model->outputDimension) = std::sqrt(colloc->weight)*(model->Operator(colloc->x, coefficients, support->GetBasisFunctions()) - model->RightHandSide(colloc->x));
	cnt += model->outputDimension;
      }
    }
    EXPECT_EQ(computedCost.size(), trueCost.size());
    EXPECT_NEAR((trueCost-computedCost).norm(), 0.0, 1.0e-12);

    for( std::size_t j=0; j<cost->numPenaltyFunctions; ++j ) {
      const Eigen::MatrixXd jacFD = cost->PenaltyFunctionJacobianByFD(j, coefficients);
      const Eigen::MatrixXd jac = cost->PenaltyFunctionJacobian(j, coefficients);
      EXPECT_NEAR((jac-jacFD).norm(), 0.0, 1.0e-6);
    }
  }
}

TEST_F(CollocationCostTests, MinimizeCost_LevenbergMarquardt) {
  // create the collocation costs
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {
    std::cout << "SUPPORT POINT: " << i << std::endl;
    
    auto support = supportCloud->GetSupportPoint(i);

    auto cost = std::make_shared<CollocationCost>(support, collocationCloud->CollocationPerSupport(i));

    pt::ptree pt;
    pt.put("FunctionTolerance", 1.0e-9);
    pt.put("GradientTolerance", 1.0e-7);
    pt.put("InitialDamping", 1.0);
    pt.put("LinearSolver", "QR");
    pt.put("MaximumFunctionEvaluations", 100000);
    pt.put("MaximumJacobianEvaluations", 100000);
    pt.put("MaxLineSearchSteps", 10);
    auto lm = std::make_shared<DenseLevenbergMarquardt>(cost, pt);

    // choose the vector of coefficients
    Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(support->NumCoefficients());

    const std::pair<Optimization::Convergence, double> info = lm->Minimize(coefficients);

    if( info.first<=0 ) { 
      std::cout << "num colloc per support i: " << collocationCloud->NumCollocationPerSupport(i) << std::endl;
      std::cout << "num unknowns: " << support->NumCoefficients() << std::endl;
      std::cout << info.first << std::endl;
      std::cout << info.second << std::endl;

      std::cout << std::endl;

      assert(false);
    }

  }

  EXPECT_TRUE(false);
    /*auto collocationCloud = std::make_shared<CollocationPointCloud>(sampler, supportCloud, cloudOptions);

  // create the collocation cost
  cost = std::make_shared<CollocationCost>(collocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*supportCloud->NumPoints());

  Eigen::MatrixXd data(outdim, supportCloud->NumPoints());
  for( std::size_t i=0; i<data.cols(); ++i ) { data.col(i) = supportCloud->GetSupportPoint(i)->x.head(outdim); }

  cost->ComputeOptimalCoefficients(data);

  for( auto it=supportCloud->Begin(); it!=supportCloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    EXPECT_TRUE(point);

    const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
    EXPECT_NEAR((x.head(outdim)-point->EvaluateLocalFunction(x)).norm(), 0.0, 1.0e-8);
    }*/
}
  /*
TEST_F(CollocationCostTests, CostFunctionEvaluation) {
  // the number of collocation points
  const std::size_t nCollocPoints = 25;

  // options for the cost function
  cloudOptions.put("NumCollocationPoints", nCollocPoints);

  auto collocationCloud = std::make_shared<CollocationPointCloud>(sampler, supportCloud, cloudOptions);

  // sample the colocation points
  collocationCloud->Resample();

  // create the colocation cost
  cost = std::make_shared<CollocationCost>(collocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*nCollocPoints);

  Eigen::MatrixXd data(outdim, supportCloud->NumPoints());
  for( std::size_t i=0; i<data.cols(); ++i ) { data.col(i) = supportCloud->GetSupportPoint(i)->x.head(outdim); }

  const Eigen::VectorXd costEval = cost->CostVector(Eigen::Map<const Eigen::VectorXd>(data.data(), data.size()));

  EXPECT_EQ(costEval.size(), cost->numPenaltyFunctions);
  for( std::size_t i=0; i<collocationCloud->numCollocationPoints; ++i ) {
    auto it = collocationCloud->GetCollocationPoint(i);
    EXPECT_TRUE(it);
    EXPECT_NEAR((it->Operator() - it->RightHandSide() - costEval.segment(i*model->outputDimension, model->outputDimension)).norm(), 0.0, 1.0e-10);
  }

  Eigen::MatrixXd jacFD(cost->numPenaltyFunctions, cost->inputDimension);

  Eigen::SparseMatrix<double> jac;
  cost->Jacobian(Eigen::Map<const Eigen::VectorXd>(data.data(), data.size()), jac);
  {
    const double dc = 1.0e-8;
    Eigen::Map<const Eigen::VectorXd> dataVec(data.data(), data.size());
    for( std::size_t c=0; c<data.size(); ++c ) {
      Eigen::VectorXd datap = dataVec;
      datap(c) += dc;
      Eigen::VectorXd datam = dataVec;
      datam(c) -= dc;
      Eigen::VectorXd data2m = dataVec;
      data2m(c) -= 2.0*dc;

      const Eigen::VectorXd costp = cost->CostVector(datap);
      const Eigen::VectorXd costm = cost->CostVector(datam);
      const Eigen::VectorXd cost2m = cost->CostVector(data2m);

      jacFD.col(c) = (2.0*costp + 3.0*costEval - 6.0*costm + cost2m)/(6.0*dc);
    }
  }

  EXPECT_EQ(jac.rows(), jacFD.rows());
  EXPECT_EQ(jac.cols(), jacFD.cols());
  for( std::size_t i=0; i<jac.rows(); ++i ) {
    for( std::size_t j=0; j<jac.cols(); ++j ) {
      EXPECT_NEAR(jac.coeff(i, j), jacFD(i, j), 1.0e-4);
    }
  }
}

TEST_F(CollocationCostTests, CostFunctionMinimization) {
  // the number of collocation points
  const std::size_t nCollocPoints = 100000;

  // options for the cost function
  cloudOptions.put("NumCollocationPoints", nCollocPoints);

  auto collocationCloud = std::make_shared<CollocationPointCloud>(sampler, supportCloud, cloudOptions);

  // sample the colocation points
  collocationCloud->Resample();

  // create the colocation cost
  cost = std::make_shared<CollocationCost>(collocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*nCollocPoints);

  pt::ptree pt;
  auto lm = std::make_shared<SparseLevenbergMarquardt>(cost, pt);
  Eigen::VectorXd costVec;
  Eigen::VectorXd data(outdim*supportCloud->NumPoints());
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {

    data.segment(i*outdim, outdim) = Eigen::VectorXd::Constant(outdim, supportCloud->GetSupportPoint(i)->x(1));
  }
  //std::cout << "TEST: " << cost->CostVector(data).transpose() << std::endl;
  lm->Minimize(data, costVec);
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  //for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {
  for( std::size_t i=0; i<std::min((std::size_t)10, collocationCloud->numCollocationPoints); ++i ) {
    //const Eigen::VectorXd pnt = dist->Sample();
    //const Eigen::VectorXd pnt = supportCloud->GetSupportPoint(i)->x;
    const Eigen::VectorXd pnt = collocationCloud->GetCollocationPoint(i)->x;
    auto support = collocationCloud->GetCollocationPoint(i)->supportPoint.lock();
    std::cout << "x: " << pnt.transpose() << std::endl;
    std::cout << "||x||: " << pnt.norm() << std::endl;
    std::cout << "x.x: " << pnt.dot(pnt) << std::endl;
    std::cout << "f(x): " << support->RightHandSide(pnt).transpose() << std::endl;
    std::cout << "u(x): " << support->EvaluateLocalFunction(pnt).transpose() << std::endl;
    std::cout << "L(u(x)): " << support->Operator(pnt).transpose() << std::endl;
    std::cout << std::endl;
  }
}
*/

} // namespace tests
} // namespace clf
