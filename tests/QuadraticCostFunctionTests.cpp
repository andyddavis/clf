#include <gtest/gtest.h>

#include "clf/QuadraticCostFunction.hpp"

#include "TestCostFunctions.hpp"

namespace clf {
namespace tests {

/// A class that runs the tests for clf::QuadraticCostFunction
class QuadraticCostFunctionTests : public::testing::Test {
protected:

  /// Test that the cost function is correct
  /**
  @param[in] cost The cost function we are testing
  */
  template<typename MATTYPE>
  void TestCostFunction(std::shared_ptr<QuadraticCostFunction<MATTYPE> > const& cost) const {
    EXPECT_EQ(cost->inputDimension, 3);
    EXPECT_EQ(cost->numPenaltyFunctions, 4);
    EXPECT_EQ(cost->numPenaltyTerms, 8);

    const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inputDimension);

    // check the penalty function implementation
    const Eigen::VectorXd f0 = cost->PenaltyFunction(0, beta);
    EXPECT_NEAR(f0(0), beta(0), 1.0e-14);
    EXPECT_NEAR(f0(1), 2.0*beta(0) + beta(2), 1.0e-14);
    const Eigen::VectorXd f1 = cost->PenaltyFunction(1, beta);
    EXPECT_NEAR(f1(0), 1.0-beta(1), 1.0e-14);
    EXPECT_NEAR(f1(1), 5.0*beta(1)-5.0, 1.0e-14);
    const Eigen::VectorXd f2 = cost->PenaltyFunction(2, beta);
    EXPECT_NEAR(f2(0), beta(2)+beta(1)-1.0, 1.0e-14);
    EXPECT_NEAR(f2(1), beta(2), 1.0e-14);
    const Eigen::VectorXd f3 = cost->PenaltyFunction(3, beta);
    EXPECT_NEAR(f3(0), 3.0*beta(2), 1.0e-14);
    EXPECT_NEAR(f3(1), beta(2)-beta(0), 1.0e-14);

    // compute the gradient with finite difference
    const Eigen::MatrixXd jac0 = cost->PenaltyFunctionJacobian(0);
    const Eigen::MatrixXd jac0beta = cost->CostFunction<MATTYPE>::PenaltyFunctionJacobian(0, beta);
    EXPECT_NEAR((jac0-jac0beta).norm(), 0.0, 1.0e-12);
    const Eigen::MatrixXd jac0FD = cost->PenaltyFunctionJacobianByFD(0, beta);
    EXPECT_NEAR((jac0-jac0FD).norm(), 0.0, 1.0e-7);

    const Eigen::MatrixXd jac1 = cost->PenaltyFunctionJacobian(1);
    const Eigen::MatrixXd jac1beta = cost->CostFunction<MATTYPE>::PenaltyFunctionJacobian(1, beta);
    EXPECT_NEAR((jac1-jac1beta).norm(), 0.0, 1.0e-12);
    const Eigen::MatrixXd jac1FD = cost->PenaltyFunctionJacobianByFD(1, beta);
    EXPECT_NEAR((jac1-jac1FD).norm(), 0.0, 1.0e-7);

    const Eigen::MatrixXd jac2 = cost->PenaltyFunctionJacobian(2);
    const Eigen::MatrixXd jac2beta = cost->CostFunction<MATTYPE>::PenaltyFunctionJacobian(2, beta);
    EXPECT_NEAR((jac2-jac2beta).norm(), 0.0, 1.0e-12);
    const Eigen::MatrixXd jac2FD = cost->PenaltyFunctionJacobianByFD(2, beta);
    EXPECT_NEAR((jac2-jac2FD).norm(), 0.0, 1.0e-7);

    const Eigen::MatrixXd jac3 = cost->PenaltyFunctionJacobian(3);
    const Eigen::MatrixXd jac3beta = cost->CostFunction<MATTYPE>::PenaltyFunctionJacobian(3, beta);
    EXPECT_NEAR((jac3-jac3beta).norm(), 0.0, 1.0e-12);
    const Eigen::MatrixXd jac3FD = cost->PenaltyFunctionJacobianByFD(3, beta);
    EXPECT_NEAR((jac3-jac3FD).norm(), 0.0, 1.0e-7);

    MATTYPE jacbeta;
    cost->Jacobian(beta, jacbeta);
    EXPECT_EQ(jacbeta.rows(), cost->numPenaltyTerms);
    EXPECT_EQ(jacbeta.cols(), cost->inputDimension);

    MATTYPE jac;
    cost->Jacobian(jac);
    EXPECT_EQ(jac.rows(), cost->numPenaltyTerms);
    EXPECT_EQ(jac.cols(), cost->inputDimension);

    Eigen::MatrixXd expectedJac = Eigen::MatrixXd::Zero(cost->numPenaltyTerms, cost->inputDimension);
    expectedJac(0, 0) = 1.0;
    expectedJac(1, 0) = 2.0;
    expectedJac(1, 2) = 1.0;
    expectedJac(2, 1) = -1.0;
    expectedJac(3, 1) = 5.0;
    expectedJac(4, 1) = 1.0;
    expectedJac(4, 2) = 1.0;
    expectedJac(5, 2) = 1.0;
    expectedJac(6, 2) = 3.0;
    expectedJac(7, 0) = -1.0;
    expectedJac(7, 2) = 1.0;

    EXPECT_NEAR(((Eigen::MatrixXd)jacbeta-expectedJac).norm(), 0.0, 1.0e-14);
    EXPECT_NEAR(((Eigen::MatrixXd)jac-expectedJac).norm(), 0.0, 1.0e-14);
  }
};

TEST_F(QuadraticCostFunctionTests, Dense) {
  auto cost = std::make_shared<tests::DenseQuadraticCostTest>();
  TestCostFunction<Eigen::MatrixXd>(cost);
}

TEST_F(QuadraticCostFunctionTests, Sparse) {
  auto cost = std::make_shared<tests::SparseQuadraticCostTest>();
  TestCostFunction<Eigen::SparseMatrix<double> >(cost);
}

} // namespace tests
} // namespace clf
