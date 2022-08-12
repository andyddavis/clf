#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"
#include "clf/LinearModel.hpp"

using namespace clf;

namespace clf {
namespace tests {

/// A class to run the tests for clf::LinearModel
class LinearModelTests : public::testing::Test {
protected:

  /// Tear down the tests
  virtual void TearDown() override {
    EXPECT_EQ(linsys->indim, indim);
    EXPECT_EQ(linsys->outdim, outdim);

    const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
    
    const Eigen::VectorXd rhs = linsys->RightHandSide(x);
    EXPECT_EQ(rhs.size(), outdim);
    EXPECT_NEAR(rhs.norm(), 0.0, 1.0e-14);
    
    const Eigen::MatrixXd A = linsys->Operator(x);
    EXPECT_EQ(A.rows(), outdim);
    EXPECT_EQ(A.cols(), matdim);
    EXPECT_NEAR((A-mat).norm(), 0.0, 1.0e-14);
    
    const std::size_t maxOrder = 4;
    std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
    
    const double delta = 0.75;
    const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
    auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
    const Eigen::VectorXd y = domain->MapToHypercube(x);
    
    auto basis = std::make_shared<LegendrePolynomials>();
    auto vec = std::make_shared<FeatureVector>(set, basis);
    auto mat = std::make_shared<FeatureMatrix>(vec, matdim, domain);
    auto func = std::make_shared<LocalFunction>(mat);
    
    const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());
    
    const Eigen::VectorXd eval = func->Evaluate(x, coeff);
    const Eigen::VectorXd op = linsys->Operator(func, x, coeff);
    EXPECT_EQ(op.size(), outdim);
    EXPECT_NEAR((A*eval-op).norm(), 0.0, 1.0e-14);
    
    const Eigen::MatrixXd jac = linsys->JacobianWRTCoefficients(func, x, coeff);
    EXPECT_EQ(jac.rows(), outdim);
    EXPECT_EQ(jac.cols(), coeff.size());
    const Eigen::MatrixXd jacFD = linsys->JacobianWRTCoefficientsFD(func, x, coeff);
    EXPECT_EQ(jacFD.rows(), outdim);
    EXPECT_EQ(jacFD.cols(), coeff.size());
    EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-12);
    
    const Eigen::VectorXd weights = Eigen::VectorXd::Random(outdim);
    const Eigen::MatrixXd hess = linsys->HessianWRTCoefficients(func, x, coeff, weights);
    EXPECT_EQ(hess.rows(), coeff.size());
    EXPECT_EQ(hess.cols(), coeff.size());
    EXPECT_NEAR(hess.norm(), 0.0, 1.0e-12);
    const Eigen::MatrixXd hessFD = linsys->HessianWRTCoefficientsFD(func, x, coeff, weights);
    EXPECT_EQ(hessFD.rows(), coeff.size());
    EXPECT_EQ(hessFD.cols(), coeff.size());
    EXPECT_NEAR(hessFD.norm(), 0.0, 1.0e-7);
  }
  
  /// The input dimension
  const std::size_t indim = 4; 

  /// The output dimension
  const std::size_t outdim = 3;

  /// The number of columns in the matrix
  std::size_t matdim;

  /// The matrix 
  Eigen::MatrixXd mat;

  /// The linear model 
  std::shared_ptr<LinearModel> linsys;
};

TEST_F(LinearModelTests, SquareIdentity) {
  matdim = outdim;
  mat = Eigen::MatrixXd::Identity(outdim, matdim);
  linsys = std::make_shared<LinearModel>(indim, outdim);
}

TEST_F(LinearModelTests, NonSquareIdentity) {
  matdim = 8;
  mat = Eigen::MatrixXd::Identity(outdim, matdim);
  linsys = std::make_shared<LinearModel>(indim, outdim, matdim);
}

TEST_F(LinearModelTests, RandomMatrix) {
  matdim = 8;
  mat = Eigen::MatrixXd::Random(outdim, matdim);
  linsys = std::make_shared<LinearModel>(indim, mat);
}

} // namespace tests
} // namespace clf
