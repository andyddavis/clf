#include <gtest/gtest.h>

#include <Eigen/Sparse>

#include "clf/LevenbergMarquardt.hpp"

#include "TestCostFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class LevenbergMarquardtTests : public::testing::Test {
protected:
  /// Test the optimization
  template<typename OPTTYPE, typename COSTTYPE>
  void TestOptimizer(std::shared_ptr<OPTTYPE> const& lm, std::shared_ptr<COSTTYPE> const& cost) {
    EXPECT_EQ(cost->inputDimension, 3);
    EXPECT_EQ(cost->numPenaltyFunctions, 4);

    EXPECT_EQ(lm->maxEvals, 1000);
    EXPECT_EQ(lm->maxJacEvals, 1000);
    EXPECT_EQ(lm->maxIters, 1000);
    EXPECT_NEAR(lm->gradTol, 1.0e-10, 1.0e-14);
    EXPECT_NEAR(lm->funcTol, 1.0e-10, 1.0e-14);
    EXPECT_NEAR(lm->betaTol, 1.0e-10, 1.0e-14);

    Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inputDimension);
    Eigen::VectorXd costVec;
    const std::pair<Optimization::Convergence, double> info = lm->Minimize(beta, costVec);
    EXPECT_TRUE(info.first>0);
    EXPECT_NEAR(info.second, 0.0, 1.0e-10);
    const double totCost = costVec.dot(costVec);
    EXPECT_NEAR(totCost, 0.0, 1.0e-10);
    EXPECT_NEAR(totCost, info.second, 1.0e-14);

    EXPECT_NEAR(beta(0), 0.0, 2.0*std::sqrt(totCost));
    EXPECT_NEAR(beta(1), 1.0, 2.0*std::sqrt(totCost));
    EXPECT_NEAR(beta(2), 0.0, 2.0*std::sqrt(totCost));
  }
};

TEST_F(LevenbergMarquardtTests, Dense) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  pt::ptree pt;
  auto lm = std::make_shared<DenseLevenbergMarquardt>(cost, pt);
  TestOptimizer(lm, cost);
}

TEST_F(LevenbergMarquardtTests, Sparse) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  pt::ptree pt;
  auto lm = std::make_shared<SparseLevenbergMarquardt>(cost, pt);
  TestOptimizer(lm, cost);
}
