#include <gtest/gtest.h>

#include "clf/QuadraticCostOptimizer.hpp"

#include "TestCostFunctions.hpp"

namespace pt = boost::property_tree;

namespace clf {
namespace tests {

/// A class to run the tests for clf::QuadraticCostOptimizer 
class QuadraticCostOptimizerTests : public::testing::Test {
protected:
  /// Make sure everything is what we expect
  template<typename COSTTYPE, typename OPTTYPE>
  void Check(std::shared_ptr<COSTTYPE> const& cost, std::shared_ptr<OPTTYPE> const& opt) {
    Eigen::VectorXd beta;
    opt->Minimize(beta);
    EXPECT_EQ(beta.size(), cost->inputDimension);
    EXPECT_NEAR(beta(0), 0.0, 1.0e-14);
    EXPECT_NEAR(beta(1), 1.0, 1.0e-14);
    EXPECT_NEAR(beta(2), 0.0, 1.0e-14);  

    const Eigen::VectorXd val = cost->CostVector(beta);
    EXPECT_NEAR(val.norm(), 0.0, 1.0e-14);
  }
};

TEST_F(QuadraticCostOptimizerTests, DenseLU) {
  auto cost = std::make_shared<tests::DenseQuadraticCostTest>();

  pt::ptree pt;
  auto opt = std::make_shared<DenseQuadraticCostOptimizer>(cost, pt);
  Check(cost, opt);
}

TEST_F(QuadraticCostOptimizerTests, DenseQR) {
  auto cost = std::make_shared<tests::DenseQuadraticCostTest>();

  pt::ptree pt;
  pt.put("LinearSolver", "QR");
  auto opt = std::make_shared<DenseQuadraticCostOptimizer>(cost, pt);
  Check(cost, opt);
}

TEST_F(QuadraticCostOptimizerTests, SparseLU) {
  auto cost = std::make_shared<tests::SparseQuadraticCostTest>();

  pt::ptree pt;
  auto opt = std::make_shared<SparseQuadraticCostOptimizer>(cost, pt);
  Check(cost, opt);
}

TEST_F(QuadraticCostOptimizerTests, SparseQR) {
  auto cost = std::make_shared<tests::SparseQuadraticCostTest>();

  pt::ptree pt;
  pt.put("LinearSolver", "QR");
  auto opt = std::make_shared<SparseQuadraticCostOptimizer>(cost, pt);
  Check(cost, opt);
}

} // namespace tests 
} // namespace clf
