#include <gtest/gtest.h>

#include "clf/Optimizer.hpp"
#include "clf/LevenbergMarquardt.hpp"
#include "clf/NLoptOptimizer.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(OptimizerTests, ConstructLevenbergMarquardt_Dense) {
  pt::ptree options;
  options.put("Method", "LevenbergMarquardt");
  auto opt = Optimizer<Eigen::MatrixXd>::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto lm = std::dynamic_pointer_cast<DenseLevenbergMarquardt>(opt);
  EXPECT_TRUE(lm);
}

TEST(OptimizerTests, ConstructNLopt_Dense) {
  pt::ptree options;
  options.put("Method", "NLopt");
  auto opt = Optimizer<Eigen::MatrixXd>::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto nl = std::dynamic_pointer_cast<DenseNLoptOptimizer>(opt);
  EXPECT_TRUE(nl);
}

TEST(OptimizerTests, ConstructLevenbergMarquardt_Sparse) {
  pt::ptree options;
  options.put("Method", "LevenbergMarquardt");
  auto opt = Optimizer<Eigen::SparseMatrix<double> >::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto lm = std::dynamic_pointer_cast<SparseLevenbergMarquardt>(opt);
  EXPECT_TRUE(lm);
}

TEST(OptimizerTests, ConstructNLopt_Sparse) {
  pt::ptree options;
  options.put("Method", "NLopt");
  auto opt = Optimizer<Eigen::SparseMatrix<double> >::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto nl = std::dynamic_pointer_cast<SparseNLoptOptimizer>(opt);
  EXPECT_TRUE(nl);
}

TEST(OptimizerTests, FailedConstruction_Dense) {
  const std::string name = "BadOptimizerNameFails";

  pt::ptree options;
  options.put("Method", name);
  try {
    auto opt = Optimizer<Eigen::MatrixXd>::Construct(nullptr, options);
  } catch( exceptions::OptimizerNameException<Optimizer<Eigen::MatrixXd> > const& exc ) {
    EXPECT_EQ(exc.name, name);
  }
}

TEST(OptimizerTests, FailedConstruction_Sparse) {
  const std::string name = "BadOptimizerNameFails";

  pt::ptree options;
  options.put("Method", name);
  try {
    auto opt = Optimizer<Eigen::SparseMatrix<double> >::Construct(nullptr, options);
  } catch( exceptions::OptimizerNameException<Optimizer<Eigen::SparseMatrix<double> > > const& exc ) {
    EXPECT_EQ(exc.name, name);
  }
}
