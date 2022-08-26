#include <gtest/gtest.h>

#include "clf/DenseLevenbergMarquardt.hpp"
#include "clf/SparseLevenbergMarquardt.hpp"

#include "TestPenaltyFunctions.hpp"

using namespace clf;

TEST(LevenbergMarquardtTests, DenseMatrices) {
  auto para = std::make_shared<Parameters>();

  // create the example penalty functions
  auto func0 = std::make_shared<tests::DensePenaltyFunctionTest0>();
  auto func1 = std::make_shared<tests::DensePenaltyFunctionTest1>();
  auto cost = std::make_shared<DenseCostFunction>(DensePenaltyFunctions({func0, func1}));

  DenseLevenbergMarquardt lm(cost, para);
  EXPECT_EQ(lm.NumParameters(), cost->InputDimension());

  Eigen::VectorXd beta = Eigen::VectorXd::Random(lm.NumParameters());
  Eigen::VectorXd costVec;
  const std::pair<Optimization::Convergence, double> result = lm.Minimize(beta, costVec);
  EXPECT_TRUE(result.first>0);

  Eigen::VectorXd expected(3);
  expected << 0.0, 1.0, 0.0;
  EXPECT_NEAR((beta-expected).norm(), 0.0, 1.0e-10);
  EXPECT_NEAR(costVec.norm(), 0.0, 1.0e-10);
  EXPECT_NEAR(cost->Gradient(beta).norm(), 0.0, 1.0e-10);
}

TEST(LevenbergMarquardtTests, SparseMatrices) {
  auto para = std::make_shared<Parameters>();

  // create the example penalty functions
  auto func0 = std::make_shared<tests::SparsePenaltyFunctionTest0>();
  auto func1 = std::make_shared<tests::SparsePenaltyFunctionTest1>();
  auto cost = std::make_shared<SparseCostFunction>(SparsePenaltyFunctions({func0, func1}));

  SparseLevenbergMarquardt lm(cost, para);
  EXPECT_EQ(lm.NumParameters(), cost->InputDimension());

  Eigen::VectorXd beta = Eigen::VectorXd::Random(lm.NumParameters());
  Eigen::VectorXd costVec;
  const std::pair<Optimization::Convergence, double> result = lm.Minimize(beta, costVec);
  EXPECT_TRUE(result.first>0);

  Eigen::VectorXd expected(3);
  expected << 0.0, 1.0, 0.0;
  EXPECT_NEAR((beta-expected).norm(), 0.0, 1.0e-10);
  EXPECT_NEAR(costVec.norm(), 0.0, 1.0e-10);
  EXPECT_NEAR(cost->Gradient(beta).norm(), 0.0, 1.0e-10);
}
