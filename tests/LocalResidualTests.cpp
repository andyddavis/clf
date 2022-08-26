#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"

#include "clf/IdentityModel.hpp"
#include "clf/LinearModel.hpp"

#include "clf/LegendrePolynomials.hpp"
#include "clf/LocalResidual.hpp"

namespace clf { 
namespace tests {

/// A class to run the tests for clf::LocalResidual
class LocalResidualTests : public::testing::Test {
protected:

  /// Tear down the tests
  virtual void TearDown() override {
    const double radius = 0.1;
    const std::size_t numPoints = 100;
    
    auto para = std::make_shared<Parameters>();
    para->Add("NumLocalPoints", numPoints);
    
    const double delta = 0.75;
    const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
    auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
    
    const std::size_t maxOrder = 4;
    std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
    auto basis = std::make_shared<LegendrePolynomials>();
    auto func = std::make_shared<LocalFunction>(set, basis, domain, outdim);
    
    LocalResidual resid(func, system, para);
    EXPECT_EQ(resid.InputDimension(), func->NumCoefficients());
    EXPECT_EQ(resid.OutputDimension(), outdim*numPoints);
    EXPECT_EQ(resid.NumPoints(), numPoints);
    
    const Eigen::VectorXd coeff = Eigen::VectorXd::Random(resid.InputDimension());
    
    const Eigen::VectorXd eval = resid.Evaluate(coeff);
    EXPECT_EQ(eval.size(), resid.OutputDimension());
    std::size_t start = 0;
    for( std::size_t i=0; i<numPoints; ++i ) {
      std::shared_ptr<Point> pnt = resid.GetPoint(i);
      for( std::size_t j=0; j<indim; ++j ) { EXPECT_NEAR(pnt->x(j)-xbar(j), 0.0, delta+1.0e-14); }
      
      EXPECT_NEAR((eval.segment(start, outdim)-mat*func->Evaluate(pnt, coeff)).norm(), 0.0, 1.0e-14);
      start += outdim;
    }
    
    const Eigen::MatrixXd jac = resid.Jacobian(coeff);
    EXPECT_EQ(jac.rows(), outdim*numPoints);
    EXPECT_EQ(jac.cols(), func->NumCoefficients());
    const Eigen::MatrixXd jacFD = resid.JacobianFD(coeff);
    EXPECT_EQ(jacFD.rows(), outdim*numPoints);
    EXPECT_EQ(jacFD.cols(), func->NumCoefficients());
    EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-10);
    
    const Eigen::VectorXd weights = Eigen::VectorXd::Random(resid.OutputDimension());
    const Eigen::MatrixXd hess = resid.Hessian(coeff, weights);
    EXPECT_EQ(hess.rows(), func->NumCoefficients());
    EXPECT_EQ(hess.cols(), func->NumCoefficients());
    EXPECT_NEAR(hess.norm(), 0.0, 1.0e-10);
    const Eigen::MatrixXd hessFD = resid.HessianFD(coeff, weights);
    EXPECT_EQ(hessFD.rows(), func->NumCoefficients());
    EXPECT_EQ(hessFD.cols(), func->NumCoefficients());
    EXPECT_NEAR(hessFD.norm(), 0.0, 1.0e-10);
  }

  /// The input dimension
  const std::size_t indim = 3;

  /// The output dimension
  const std::size_t outdim = 2;

  /// The matrix that defines the model
  Eigen::MatrixXd mat;

  /// The matrix the system of equation whose residual we are checking
  std::shared_ptr<SystemOfEquations> system;

};

TEST_F(LocalResidualTests, IdentityModel) {
  mat = Eigen::MatrixXd::Identity(outdim, outdim);
  system = std::make_shared<IdentityModel>(indim, outdim);
}

TEST_F(LocalResidualTests, LinearModel) {
  mat = Eigen::MatrixXd::Random(outdim, outdim);
  system = std::make_shared<LinearModel>(indim, mat);
}

} // namespace tests
} // namespace clf

