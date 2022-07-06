#include <gtest/gtest.h>

#include "clf/LevenbergMarquardt.hpp"

#include "TestCostFunctions.hpp"

using namespace clf;

TEST(LevenbergMarquardtTests, DenseMatrices) {
  auto para = std::make_shared<Parameters>();

  std::size_t indim = 8;
  auto cost = std::make_shared<tests::DenseCostFunctionTest>(indim);

  DenseLevenbergMarquardt lm(cost, para);
  EXPECT_EQ(lm.NumParameters(), indim);

  Eigen::VectorXd beta = Eigen::VectorXd::Random(indim);
  Eigen::VectorXd costVec;
  const std::pair<Optimization::Convergence, double> result = lm.Minimize(beta, costVec);
  EXPECT_TRUE(result.first>0);
}

TEST(LevenbergMarquardtTests, SparseMatrices) {
  auto para = std::make_shared<Parameters>();

  const std::size_t indim = 13;
  auto cost = std::make_shared<tests::SparseCostFunctionTest>(indim);

  SparseLevenbergMarquardt lm(cost, para);
  EXPECT_EQ(lm.NumParameters(), indim);

  Eigen::VectorXd beta = Eigen::VectorXd::Random(indim);
  Eigen::VectorXd costVec;
  const std::pair<Optimization::Convergence, double> result = lm.Minimize(beta, costVec);
  EXPECT_TRUE(result.first>0);
}
