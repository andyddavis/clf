#include <gtest/gtest.h>

#include "clf/BoundaryCondition.hpp"
#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"
#include "clf/IdentityModel.hpp"

namespace clf {
namespace tests {

/// A class to run the tests for clf::BoundaryCondition
class BoundaryConditionTests : public::testing::Test {
protected:
};

TEST_F(BoundaryConditionTests, IdentityModel) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;

  auto superDom = std::make_shared<Hypercube>(Eigen::VectorXd::Constant(indim, -0.1), Eigen::VectorXd::Constant(indim, 0.9));
  
  // create the local function
  auto para = std::make_shared<Parameters>();
  para->Add("InputDimension", indim);
  para->Add("OutputDimension", outdim);
  auto dom = std::make_shared<Hypercube>(indim);
  dom->SetSuperset(superDom);
  const std::size_t maxOrder = 3;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  auto leg = std::make_shared<LegendrePolynomials>();
  auto func = std::make_shared<LocalFunction>(set, leg, dom, para);

  // create a system of equations
  auto system = std::make_shared<IdentityModel>(indim, outdim);

  // create a boundary condition cost function
  auto bc = std::make_shared<BoundaryCondition>(func, system);
  EXPECT_EQ(bc->InputDimension(), func->NumCoefficients());
  EXPECT_EQ(bc->OutputDimension(), 0);

  // add random points from the boundary of the super set
  std::size_t numPoints = 25;
  for( std::size_t i=0; i<numPoints; ++i ) {
    bc->AddPoint(superDom->SampleBoundary());
    EXPECT_EQ(bc->NumPoints(), i+1);
    EXPECT_EQ(bc->OutputDimension(), (i+1)*outdim);
  }

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(bc->InputDimension());
    
  const Eigen::VectorXd eval = bc->Evaluate(coeff);
  EXPECT_EQ(eval.size(), bc->OutputDimension());
  std::size_t start = 0;
  for( std::size_t i=0; i<numPoints; ++i ) {
    std::shared_ptr<Point> pnt = bc->GetPoint(i);
    
    EXPECT_NEAR((eval.segment(start, outdim)-func->Evaluate(pnt, coeff)).norm(), 0.0, 1.0e-14);
    start += outdim;
  }
  
  const Eigen::MatrixXd jac = bc->Jacobian(coeff);
  EXPECT_EQ(jac.rows(), outdim*numPoints);
  EXPECT_EQ(jac.cols(), func->NumCoefficients());
  const Eigen::MatrixXd jacFD = bc->JacobianFD(coeff);
  EXPECT_EQ(jacFD.rows(), outdim*numPoints);
  EXPECT_EQ(jacFD.cols(), func->NumCoefficients());
  EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-10);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(bc->OutputDimension());
  const Eigen::MatrixXd hess = bc->Hessian(coeff, weights);
  EXPECT_EQ(hess.rows(), func->NumCoefficients());
  EXPECT_EQ(hess.cols(), func->NumCoefficients());
  EXPECT_NEAR(hess.norm(), 0.0, 1.0e-10);
  const Eigen::MatrixXd hessFD = bc->HessianFD(coeff, weights);
  EXPECT_EQ(hessFD.rows(), func->NumCoefficients());
  EXPECT_EQ(hessFD.cols(), func->NumCoefficients());
  EXPECT_NEAR(hessFD.norm(), 0.0, 1.0e-10);
}
  
} // namespace tests
} // namespace clf
