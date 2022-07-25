#include <gtest/gtest.h>

#include "clf/LinearSystem.hpp"
#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(LinearSystemTests, SquareIdentity) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  LinearSystem linsys(indim, outdim);
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
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = linsys.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_EQ(op.size(), eval.size());
  EXPECT_NEAR((eval-op).norm(), 0.0, 1.0e-14);
}

TEST(LinearSystemTests, NonSquareIdentity) {
  const std::size_t indim = 5;
  const std::size_t matdim = 8;
  const std::size_t outdim = 3;
  LinearSystem linsys(indim, outdim, matdim);
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
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, matdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = linsys.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_NEAR((A*eval-op).norm(), 0.0, 1.0e-14);
}

TEST(LinearSystemTests, RandomMatrix) {
  const std::size_t indim = 5;
  const std::size_t matdim = 8;
  const std::size_t outdim = 3;
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(outdim, matdim);
  LinearSystem linsys(indim, A);
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
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, matdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = linsys.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_NEAR((A*eval-op).norm(), 0.0, 1.0e-14);
}
