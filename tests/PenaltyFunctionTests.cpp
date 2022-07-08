#include <gtest/gtest.h>

#include "TestPenaltyFunctions.hpp"

using namespace clf;

TEST(PenaltyFunctionTests, DenseTest0) {
  // create the example penalty function
  tests::DensePenaltyFunctionTest0 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.indim, 3);
  EXPECT_EQ(func.outdim, 2);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.outdim);
  EXPECT_NEAR(eval(0), beta(0), 1.0e-14);
  EXPECT_NEAR(eval(1), beta(0)*(1.0-beta(2)), 1.0e-14);
}

TEST(PenaltyFunctionTests, DenseTest1) {
  // create the example penalty function
  tests::DensePenaltyFunctionTest1 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.indim, 3);
  EXPECT_EQ(func.outdim, 6);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.outdim);
  EXPECT_NEAR(eval(0), 1.0-beta(1), 1.0e-14);
  EXPECT_NEAR(eval(1), 1.0-beta(1)+beta(2), 1.0e-14);
  EXPECT_NEAR(eval(2), beta(2), 1.0e-14);
  EXPECT_NEAR(eval(3), beta(2)*(1.0-beta(1)), 1.0e-14);
  EXPECT_NEAR(eval(4), beta(0)*beta(2), 1.0e-14);
  EXPECT_NEAR(eval(5), beta(0)*beta(0)*beta(1), 1.0e-14);
}

TEST(PenaltyFunctionTests, SparseTest0) {
  // create the example penalty function
  tests::SparsePenaltyFunctionTest0 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.indim, 3);
  EXPECT_EQ(func.outdim, 2);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.outdim);
  EXPECT_NEAR(eval(0), beta(0), 1.0e-14);
  EXPECT_NEAR(eval(1), beta(0)*(1.0-beta(2)), 1.0e-14);
}

TEST(PenaltyFunctionTests, SparseTest1) {
  // create the example penalty function
  tests::SparsePenaltyFunctionTest1 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.indim, 3);
  EXPECT_EQ(func.outdim, 6);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.outdim);
  EXPECT_NEAR(eval(0), 1.0-beta(1), 1.0e-14);
  EXPECT_NEAR(eval(1), 1.0-beta(1)+beta(2), 1.0e-14);
  EXPECT_NEAR(eval(2), beta(2), 1.0e-14);
  EXPECT_NEAR(eval(3), beta(2)*(1.0-beta(1)), 1.0e-14);
  EXPECT_NEAR(eval(4), beta(0)*beta(2), 1.0e-14);
  EXPECT_NEAR(eval(5), beta(0)*beta(0)*beta(1), 1.0e-14);
}
