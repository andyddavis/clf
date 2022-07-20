#include <gtest/gtest.h>

#include "clf/CostFunction.hpp"

#include "TestPenaltyFunctions.hpp"

using namespace clf;

TEST(CostFunctionTests, DenseMatrices) {
  // create the example penalty functions
  auto func0 = std::make_shared<tests::DensePenaltyFunctionTest0>();
  auto func1 = std::make_shared<tests::DensePenaltyFunctionTest1>();

  DenseCostFunction cost({func0, func1});
  EXPECT_EQ(cost.InputDimension(), func0->indim);
  EXPECT_EQ(cost.InputDimension(), func1->indim);
  EXPECT_EQ(cost.numPenaltyFunctions, 2);
  EXPECT_EQ(cost.numTerms, func0->outdim+func1->outdim);

  const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost.InputDimension());
  const Eigen::VectorXd eval = cost.Evaluate(beta);
  EXPECT_EQ(eval.size(), cost.numTerms);

  Eigen::VectorXd expected(8);
  expected << beta(0), beta(0)*(1.0-beta(2)), 1.0-beta(1), 1.0-beta(1)+beta(2), beta(2), beta(2)*(1.0-beta(1)), beta(0)*beta(2), beta(0)*beta(0)*beta(1);
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);

  Eigen::MatrixXd jac;
  cost.Jacobian(beta, jac);
  EXPECT_EQ(jac.cols(), func0->indim);
  EXPECT_EQ(jac.cols(), func1->indim);
  EXPECT_EQ(jac.rows(), func0->outdim+func1->outdim);
  EXPECT_NEAR((jac.topLeftCorner(func0->outdim, func0->indim)-func0->Jacobian(beta)).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((jac.bottomLeftCorner(func1->outdim, func1->indim)-func1->Jacobian(beta)).norm(), 0.0, 1.0e-14);
}

TEST(CostFunctionTests, SparseMatrices) {
  // create the example penalty functions
  auto func0 = std::make_shared<tests::SparsePenaltyFunctionTest0>();
  auto func1 = std::make_shared<tests::SparsePenaltyFunctionTest1>();

  SparseCostFunction cost({func0, func1});
  EXPECT_EQ(cost.InputDimension(), func0->indim);
  EXPECT_EQ(cost.InputDimension(), func1->indim);
  EXPECT_EQ(cost.numPenaltyFunctions, 2);
  EXPECT_EQ(cost.numTerms, func0->outdim+func1->outdim);

  const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost.InputDimension());
  const Eigen::VectorXd eval = cost.Evaluate(beta);
  EXPECT_EQ(eval.size(), cost.numTerms);

  Eigen::VectorXd expected(8);
  expected << beta(0), beta(0)*(1.0-beta(2)), 1.0-beta(1), 1.0-beta(1)+beta(2), beta(2), beta(2)*(1.0-beta(1)), beta(0)*beta(2), beta(0)*beta(0)*beta(1);
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);

  Eigen::SparseMatrix<double> jac;
  cost.Jacobian(beta, jac);
  EXPECT_EQ(jac.cols(), func0->indim);
  EXPECT_EQ(jac.cols(), func1->indim);
  EXPECT_EQ(jac.rows(), func0->outdim+func1->outdim);
  EXPECT_NEAR((jac.topLeftCorner(func0->outdim, func0->indim)-func0->Jacobian(beta)).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((jac.bottomLeftCorner(func1->outdim, func1->indim)-func1->Jacobian(beta)).norm(), 0.0, 1.0e-14);
}
