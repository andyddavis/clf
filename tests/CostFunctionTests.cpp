#include <gtest/gtest.h>

#include "clf/CostFunction.hpp"

#include "TestCostFunctions.hpp"

namespace clf {
namespace tests {

/// A class that runs the tests for clf::CostFunction
class CostFunctionTests : public::testing::Test {
protected:
  /// Test that the cost function is correct
  /**
  @param[in] cost The cost function we are testing
   */
  template<typename MATTYPE>
  void TestCostFunction(std::shared_ptr<CostFunction<MATTYPE> > const& cost) const {
    EXPECT_EQ(cost->inputDimension, 3);
    EXPECT_EQ(cost->numPenaltyFunctions, 4);

    const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inputDimension);

    // check the penalty function implementation
    EXPECT_NEAR(cost->PenaltyFunction(0, beta), beta(0), 1.0e-14);
    EXPECT_NEAR(cost->PenaltyFunction(1, beta), 1.0-beta(1), 1.0e-14);
    EXPECT_NEAR(cost->PenaltyFunction(2, beta), beta(2), 1.0e-14);
    EXPECT_NEAR(cost->PenaltyFunction(3, beta), beta(0)*beta(2), 1.0e-14);

    // check the cost vector
    const Eigen::VectorXd costVec = cost->CostVector(beta);
    EXPECT_EQ(costVec.size(), cost->numPenaltyFunctions);
    EXPECT_NEAR(costVec(0), beta(0), 1.0e-14);
    EXPECT_NEAR(costVec(1), 1.0-beta(1), 1.0e-14);
    EXPECT_NEAR(costVec(2), beta(2), 1.0e-14);
    EXPECT_NEAR(costVec(3), beta(0)*beta(2), 1.0e-14);

    // compute the gradient with finite difference
    const Eigen::VectorXd grad0 = cost->PenaltyFunctionGradient(0, beta);
    const Eigen::VectorXd grad0FD = cost->PenaltyFunctionGradientByFD(0, beta);
    EXPECT_NEAR((grad0-grad0FD).norm(), 0.0, 1.0e-8);

    const Eigen::VectorXd grad1 = cost->PenaltyFunctionGradient(1, beta);
    const Eigen::VectorXd grad1FD = cost->PenaltyFunctionGradientByFD(1, beta);
    EXPECT_NEAR((grad1-grad1FD).norm(), 0.0, 1.0e-7);

    const Eigen::VectorXd grad2 = cost->PenaltyFunctionGradient(2, beta);
    const Eigen::VectorXd grad2FD = cost->PenaltyFunctionGradientByFD(2, beta);
    EXPECT_NEAR((grad2-grad2FD).norm(), 0.0, 1.0e-8);

    const Eigen::VectorXd grad3 = cost->PenaltyFunctionGradient(3, beta);
    const Eigen::VectorXd grad3FD = cost->PenaltyFunctionGradientByFD(3, beta);
    EXPECT_NEAR((grad3-grad3FD).norm(), 0.0, 1.0e-7);

    MATTYPE jac;
    cost->Jacobian(beta, jac);
    EXPECT_EQ(jac.rows(), cost->numPenaltyFunctions);
    EXPECT_EQ(jac.cols(), cost->inputDimension);

    Eigen::MatrixXd expectedJac = Eigen::MatrixXd::Zero(cost->numPenaltyFunctions, cost->inputDimension);
    expectedJac(0, 0) = 1.0;
    expectedJac(1, 1) = -1.0;
    expectedJac(2, 2) = 1.0;
    expectedJac(3, 0) = beta(2);
    expectedJac(3, 2) = beta(0);

    EXPECT_NEAR(((Eigen::MatrixXd)jac-expectedJac).norm(), 0.0, 1.0e-14);
  }
};

TEST_F(CostFunctionTests, Dense) {
  auto cost = std::make_shared<tests::DenseCostTest>();
  TestCostFunction<Eigen::MatrixXd>(cost);

  const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inputDimension);
}

TEST_F(CostFunctionTests, Sparse) {
  auto cost = std::make_shared<tests::SparseCostTest>();
  TestCostFunction<Eigen::SparseMatrix<double> >(cost);
}

} // namespace tests
} // namespace clf
