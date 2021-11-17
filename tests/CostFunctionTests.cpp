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
    EXPECT_EQ(cost->numPenaltyTerms, 8);
    for( std::size_t i=0; i<4; ++i ) { EXPECT_EQ(cost->PenaltyFunctionOutputDimension(i), 2); }

    const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inputDimension);

    // check the penalty function implementation
    const Eigen::VectorXd f0 = cost->PenaltyFunction(0, beta);
    EXPECT_EQ(f0.size(), 2);
    EXPECT_NEAR(f0(0), beta(0), 1.0e-14);
    EXPECT_NEAR(f0(1), beta(0)*(1.0-beta(2)), 1.0e-14);
    EXPECT_NEAR(cost->PenaltyFunction(1, beta) (0), 1.0-beta(1), 1.0e-14);
    const Eigen::VectorXd f1 = cost->PenaltyFunction(1, beta);
    EXPECT_EQ(f1.size(), 2);
    EXPECT_NEAR(f1(0), 1.0-beta(1), 1.0e-14);
    EXPECT_NEAR(f1(1), 1.0-beta(1)+beta(2), 1.0e-14);
    const Eigen::VectorXd f2 = cost->PenaltyFunction(2, beta);
    EXPECT_EQ(f2.size(), 2);
    EXPECT_NEAR(f2(0), beta(2), 1.0e-14);
    EXPECT_NEAR(f2(1), beta(2)*(1.0-beta(1)), 1.0e-14);
    const Eigen::VectorXd f3 = cost->PenaltyFunction(3, beta);
    EXPECT_EQ(f3.size(), 2);
    EXPECT_NEAR(f3(0), beta(0)*beta(2), 1.0e-14);
    EXPECT_NEAR(f3(1), beta(0)*beta(0)*beta(1), 1.0e-14);

    // check the cost vector
    const Eigen::VectorXd costVec = cost->CostVector(beta);
    EXPECT_EQ(costVec.size(), 2*cost->numPenaltyFunctions);
    EXPECT_NEAR(costVec(0), beta(0), 1.0e-14);
    EXPECT_NEAR(costVec(1), beta(0)*(1.0-beta(2)), 1.0e-14);
    EXPECT_NEAR(costVec(2), 1.0-beta(1), 1.0e-14);
    EXPECT_NEAR(costVec(3), 1.0-beta(1)+beta(2), 1.0e-14);
    EXPECT_NEAR(costVec(4), beta(2), 1.0e-14);
    EXPECT_NEAR(costVec(5), beta(2)*(1.0-beta(1)), 1.0e-14);
    EXPECT_NEAR(costVec(6), beta(0)*beta(2), 1.0e-14);
    EXPECT_NEAR(costVec(7), beta(0)*beta(0)*beta(1), 1.0e-14);

    // compute the Jacobian with finite difference
    const Eigen::MatrixXd grad0 = cost->PenaltyFunctionJacobian(0, beta);
    const Eigen::MatrixXd grad0FD_firstd = cost->PenaltyFunctionJacobianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::FIRST_DOWNWARD);
    EXPECT_NEAR((grad0-grad0FD_firstd).norm(), 0.0, 1.0e-7);
    const Eigen::MatrixXd grad0FD_firstu = cost->PenaltyFunctionJacobianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::FIRST_UPWARD);
    EXPECT_NEAR((grad0-grad0FD_firstu).norm(), 0.0, 1.0e-7);
    const Eigen::MatrixXd grad0FD_second = cost->PenaltyFunctionJacobianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::SECOND);
    EXPECT_NEAR((grad0-grad0FD_second).norm(), 0.0, 1.0e-7);
    const Eigen::MatrixXd grad0FD_fourth = cost->PenaltyFunctionJacobianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::FOURTH);
    EXPECT_NEAR((grad0-grad0FD_fourth).norm(), 0.0, 1.0e-7);
    const Eigen::MatrixXd grad0FD_sixth = cost->PenaltyFunctionJacobianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::SIXTH);
    EXPECT_NEAR((grad0-grad0FD_sixth).norm(), 0.0, 1.0e-7);

    const Eigen::MatrixXd grad1 = cost->PenaltyFunctionJacobian(1, beta);
    const Eigen::MatrixXd grad1FD = cost->PenaltyFunctionJacobianByFD(1, beta);
    EXPECT_NEAR((grad1-grad1FD).norm(), 0.0, 1.0e-7);

    const Eigen::MatrixXd grad2 = cost->PenaltyFunctionJacobian(2, beta);
    const Eigen::MatrixXd grad2FD = cost->PenaltyFunctionJacobianByFD(2, beta);
    EXPECT_NEAR((grad2-grad2FD).norm(), 0.0, 1.0e-7);

    const Eigen::MatrixXd grad3 = cost->PenaltyFunctionJacobian(3, beta);
    const Eigen::MatrixXd grad3FD = cost->PenaltyFunctionJacobianByFD(3, beta);
    EXPECT_NEAR((grad3-grad3FD).norm(), 0.0, 1.0e-7);

    // compute the Hessian with finite difference
    const std::vector<MATTYPE> hess0 = cost->PenaltyFunctionHessian(0, beta);
    const std::vector<MATTYPE> hess0FD_firstd = cost->PenaltyFunctionHessianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::FIRST_DOWNWARD);
    EXPECT_EQ(hess0.size(), hess0FD_firstd.size());
    const std::vector<MATTYPE> hess0FD_firstu = cost->PenaltyFunctionHessianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::FIRST_UPWARD);
    EXPECT_EQ(hess0.size(), hess0FD_firstu.size());
    const std::vector<MATTYPE> hess0FD_second = cost->PenaltyFunctionHessianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::SECOND);
    EXPECT_EQ(hess0.size(), hess0FD_second.size());
    const std::vector<MATTYPE> hess0FD_fourth = cost->PenaltyFunctionHessianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::FOURTH);
    EXPECT_EQ(hess0.size(), hess0FD_fourth.size());
    const std::vector<MATTYPE> hess0FD_sixth = cost->PenaltyFunctionHessianByFD(0, beta, CostFunction<MATTYPE>::FDOrder::SIXTH);
    EXPECT_EQ(hess0.size(), hess0FD_sixth.size());
    for( std::size_t i=0; i<hess0.size(); ++i ) {
      EXPECT_NEAR((hess0[i]-hess0FD_firstd[i]).norm(), 0.0, 1.0e-7);
      EXPECT_NEAR((hess0[i]-hess0FD_firstu[i]).norm(), 0.0, 1.0e-7);
      EXPECT_NEAR((hess0[i]-hess0FD_second[i]).norm(), 0.0, 1.0e-7);
      EXPECT_NEAR((hess0[i]-hess0FD_fourth[i]).norm(), 0.0, 1.0e-7);
      EXPECT_NEAR((hess0[i]-hess0FD_sixth[i]).norm(), 0.0, 1.0e-7);
    }

    const std::vector<MATTYPE> hess1 = cost->PenaltyFunctionHessian(1, beta);
    const std::vector<MATTYPE> hess1FD = cost->PenaltyFunctionHessianByFD(1, beta, CostFunction<MATTYPE>::FDOrder::SIXTH);
    EXPECT_EQ(hess1.size(), hess1FD.size());
    for( std::size_t i=0; i<hess1.size(); ++i ) { EXPECT_NEAR((hess1[i]-hess1FD[i]).norm(), 0.0, 1.0e-7); }

    const std::vector<MATTYPE> hess2 = cost->PenaltyFunctionHessian(2, beta);
    const std::vector<MATTYPE> hess2FD = cost->PenaltyFunctionHessianByFD(2, beta, CostFunction<MATTYPE>::FDOrder::SIXTH);
    EXPECT_EQ(hess2.size(), hess2FD.size());
    for( std::size_t i=0; i<hess2.size(); ++i ) { EXPECT_NEAR((hess2[i]-hess2FD[i]).norm(), 0.0, 1.0e-7); }

    const std::vector<MATTYPE> hess3 = cost->PenaltyFunctionHessian(3, beta);
    const std::vector<MATTYPE> hess3FD = cost->PenaltyFunctionHessianByFD(3, beta, CostFunction<MATTYPE>::FDOrder::SIXTH);
    EXPECT_EQ(hess3.size(), hess3FD.size());
    for( std::size_t i=0; i<hess3.size(); ++i ) { EXPECT_NEAR((hess3[i]-hess3FD[i]).norm(), 0.0, 1.0e-7); }

    MATTYPE jac;
    cost->Jacobian(beta, jac);
    EXPECT_EQ(jac.rows(), cost->numPenaltyTerms);
    EXPECT_EQ(jac.cols(), cost->inputDimension);

    Eigen::MatrixXd expectedJac = Eigen::MatrixXd::Zero(cost->numPenaltyTerms, cost->inputDimension);
    expectedJac(0, 0) = 1.0;
    expectedJac(1, 0) = 1.0-beta(2);
    expectedJac(1, 2) = -beta(0);
    expectedJac(2, 1) = -1.0;
    expectedJac(3, 1) = -1.0;
    expectedJac(3, 2) = 1.0;
    expectedJac(4, 2) = 1.0;
    expectedJac(5, 1) = -beta(2);
    expectedJac(5, 2) = 1.0-beta(1);
    expectedJac(6, 0) = beta(2);
    expectedJac(6, 2) = beta(0);
    expectedJac(7, 0) = 2.0*beta(0)*beta(1);
    expectedJac(7, 1) = beta(0)*beta(0);

    EXPECT_NEAR(((Eigen::MatrixXd)jac-expectedJac).norm(), 0.0, 1.0e-14);

    // check the Gauss-Newton Hessian 
    MATTYPE gnHess;
    cost->GaussNewtonHessian(beta, gnHess);
    EXPECT_NEAR(((Eigen::MatrixXd)gnHess-2.0*expectedJac.transpose()*expectedJac).norm(), 0.0, 1.0e-14);
    cost->GaussNewtonHessianGivenJacobian(jac, gnHess);
    EXPECT_NEAR(((Eigen::MatrixXd)gnHess-2.0*expectedJac.transpose()*expectedJac).norm(), 0.0, 1.0e-14);

    // check the full Hessian 
    MATTYPE hess;
    cost->Hessian(beta, hess);
    Eigen::MatrixXd hessFD = cost->HessianByFD(beta);
    EXPECT_NEAR(((Eigen::MatrixXd)hess-hessFD).norm(), 0.0, 1.0e-6);
    cost->Hessian(beta, jac, hess);
    EXPECT_NEAR(((Eigen::MatrixXd)hess-hessFD).norm(), 0.0, 1.0e-6);
  }
};

TEST_F(CostFunctionTests, Dense) {
  auto cost = std::make_shared<tests::DenseCostFunctionTest>();
  TestCostFunction<Eigen::MatrixXd>(cost);
}

TEST_F(CostFunctionTests, Sparse) {
  auto cost = std::make_shared<tests::SparseCostFunctionTest>();
  TestCostFunction<Eigen::SparseMatrix<double> >(cost);
}

} // namespace tests
} // namespace clf
