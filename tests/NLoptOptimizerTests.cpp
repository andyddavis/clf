#include <gtest/gtest.h>

#include "clf/NLoptOptimizer.hpp"

#include "TestCostFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

namespace clf {
namespace tests {

/// A class to run the tests for the NLopt optimizer
class NLoptOptimizerTests : public::testing::Test {
protected:

  /// Set up the options for the optimizer
  virtual void SetUp() override {
    options.put("GradientTolerance", 1.0e-11);
    options.put("FunctionTolerance", 1.0e-11);
    options.put("MaxEvaluations", 10000);
  }

  /// Test the optimization
  template<typename OPTTYPE, typename COSTTYPE>
  void TestOptimizer(std::shared_ptr<OPTTYPE> const& nlopt, std::shared_ptr<COSTTYPE> const& cost, double const tol = 1.0e-8) {
    EXPECT_EQ(cost->inputDimension, 3);
    EXPECT_EQ(cost->numPenaltyFunctions, 4);

    Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inputDimension);
    const std::pair<Optimization::Convergence, double> info = nlopt->Minimize(beta);

    EXPECT_TRUE(info.first>=0);
    EXPECT_NEAR(info.second, 0.0, tol);
    EXPECT_NEAR(beta(0), 0.0, tol);
    EXPECT_NEAR(beta(1), 1.0, tol);
    EXPECT_NEAR(beta(2), 0.0, tol);
  }

  /// Options for the optimization
  pt::ptree options;
};

TEST_F(NLoptOptimizerTests, Dense_BOBYQA) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "BOBYQA");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_NEWUOA) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "NEWUOA");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost, 1.0e-4);
}

TEST_F(NLoptOptimizerTests, Dense_PRAXIS) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "PRAXIS");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_SBPLX) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "SBPLX");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_NM) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "NM");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_MMA) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "MMA");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_SLSQP) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "SLSQP");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_PreTN) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "PreTN");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_COBYLA) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "COBYLA");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_LMVM) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "LMVM");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Dense_LBFGS) {
  auto cost = std::make_shared<tests::DenseCostTest>();

  options.put("Algorithm", "LBFGS");
  auto nlopt = std::make_shared<DenseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_BOBYQA) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "BOBYQA");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_NEWUOA) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "NEWUOA");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost, 1.0e-4);
}

TEST_F(NLoptOptimizerTests, Sparse_PRAXIS) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "PRAXIS");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_SBPLX) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "SBPLX");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_NM) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "NM");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_MMA) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "MMA");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_SLSQP) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "SLSQP");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_PreTN) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "PreTN");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_COBYLA) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "COBYLA");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost, 1.0e-2);
}

TEST_F(NLoptOptimizerTests, Sparse_LMVM) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "LMVM");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

TEST_F(NLoptOptimizerTests, Sparse_LBFGS) {
  auto cost = std::make_shared<tests::SparseCostTest>();

  options.put("Algorithm", "LBFGS");
  auto nlopt = std::make_shared<SparseNLoptOptimizer>(cost, options);
  TestOptimizer(nlopt, cost);
}

} // namespace tests
} // namespace clf
