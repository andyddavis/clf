#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(LinearModelTests, SquareIdentity) {
  const std::size_t indim = 4;
  const std::size_t outdim = 3;
  LinearModel linsys(indim, outdim);
  EXPECT_EQ(linsys.indim, indim);
  EXPECT_EQ(linsys.outdim, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  const Eigen::VectorXd rhs = linsys.RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  EXPECT_NEAR(rhs.norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd A = linsys.Operator(x);
  EXPECT_EQ(A.rows(), outdim);
  EXPECT_EQ(A.cols(), outdim);
  EXPECT_NEAR((A-Eigen::MatrixXd::Identity(outdim, outdim)).norm(), 0.0, 1.0e-14);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.1;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis, delta, xbar);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = linsys.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_EQ(op.size(), eval.size());
  EXPECT_NEAR((eval-op).norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd jac = linsys.JacobianWRTCoefficients(func, x, coeff);
  EXPECT_EQ(jac.rows(), outdim);
  EXPECT_EQ(jac.cols(), coeff.size());
  const Eigen::MatrixXd jacFD = linsys.JacobianWRTCoefficientsFD(func, x, coeff);
  EXPECT_EQ(jacFD.rows(), outdim);
  EXPECT_EQ(jacFD.cols(), coeff.size());
  EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-12);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(linsys.outdim);
  const Eigen::MatrixXd hess = linsys.HessianWRTCoefficients(func, x, coeff, weights);
  EXPECT_EQ(hess.rows(), coeff.size());
  EXPECT_EQ(hess.cols(), coeff.size());
  const Eigen::MatrixXd hessFD = linsys.HessianWRTCoefficientsFD(func, x, coeff, weights);
  EXPECT_EQ(hessFD.rows(), coeff.size());
  EXPECT_EQ(hessFD.cols(), coeff.size());
  EXPECT_NEAR((hess-hessFD).norm()/jac.norm(), 0.0, 1.0e-12);
}

TEST(LinearModelTests, NonSquareIdentity) {
  const std::size_t indim = 4;
  const std::size_t matdim = 8;
  const std::size_t outdim = 3;
  LinearModel linsys(indim, outdim, matdim);
  EXPECT_EQ(linsys.indim, indim);
  EXPECT_EQ(linsys.outdim, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  const Eigen::VectorXd rhs = linsys.RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  EXPECT_NEAR(rhs.norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd A = linsys.Operator(x);
  EXPECT_EQ(A.rows(), outdim);
  EXPECT_EQ(A.cols(), matdim);
  EXPECT_NEAR((A-Eigen::MatrixXd::Identity(outdim, matdim)).norm(), 0.0, 1.0e-14);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.1;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis, delta, xbar);
  auto mat = std::make_shared<FeatureMatrix>(vec, matdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = linsys.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_NEAR((A*eval-op).norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd jac = linsys.JacobianWRTCoefficients(func, x, coeff);
  EXPECT_EQ(jac.rows(), outdim);
  EXPECT_EQ(jac.cols(), coeff.size());
  const Eigen::MatrixXd jacFD = linsys.JacobianWRTCoefficientsFD(func, x, coeff);
  EXPECT_EQ(jacFD.rows(), outdim);
  EXPECT_EQ(jacFD.cols(), coeff.size());
  EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-12);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(linsys.outdim);
  const Eigen::MatrixXd hess = linsys.HessianWRTCoefficients(func, x, coeff, weights);
  EXPECT_EQ(hess.rows(), coeff.size());
  EXPECT_EQ(hess.cols(), coeff.size());
  const Eigen::MatrixXd hessFD = linsys.HessianWRTCoefficientsFD(func, x, coeff, weights);
  EXPECT_EQ(hessFD.rows(), coeff.size());
  EXPECT_EQ(hessFD.cols(), coeff.size());
  EXPECT_NEAR((hess-hessFD).norm()/jac.norm(), 0.0, 1.0e-12);
}

TEST(LinearModelTests, RandomMatrix) {
  const std::size_t indim = 4;
  const std::size_t matdim = 8;
  const std::size_t outdim = 3;
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(outdim, matdim);
  LinearModel linsys(indim, A);
  EXPECT_EQ(linsys.indim, indim);
  EXPECT_EQ(linsys.outdim, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  const Eigen::VectorXd rhs = linsys.RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  EXPECT_NEAR(rhs.norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd Aeval = linsys.Operator(x);
  EXPECT_EQ(Aeval.rows(), outdim);
  EXPECT_EQ(Aeval.cols(), matdim);
  EXPECT_NEAR((Aeval-A).norm(), 0.0, 1.0e-14);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.5;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis, delta, xbar);
  auto mat = std::make_shared<FeatureMatrix>(vec, matdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = linsys.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_NEAR((A*eval-op).norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd jac = linsys.JacobianWRTCoefficients(func, x, coeff);
  EXPECT_EQ(jac.rows(), outdim);
  EXPECT_EQ(jac.cols(), coeff.size());
  const Eigen::MatrixXd jacFD = linsys.JacobianWRTCoefficientsFD(func, x, coeff);
  EXPECT_EQ(jacFD.rows(), outdim);
  EXPECT_EQ(jacFD.cols(), coeff.size());
  EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-12);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(linsys.outdim);
  const Eigen::MatrixXd hess = linsys.HessianWRTCoefficients(func, x, coeff, weights);
  EXPECT_EQ(hess.rows(), coeff.size());
  EXPECT_EQ(hess.cols(), coeff.size());
  const Eigen::MatrixXd hessFD = linsys.HessianWRTCoefficientsFD(func, x, coeff, weights);
  EXPECT_EQ(hessFD.rows(), coeff.size());
  EXPECT_EQ(hessFD.cols(), coeff.size());
  EXPECT_NEAR((hess-hessFD).norm()/jac.norm(), 0.0, 1.0e-12);
}
