#include <gtest/gtest.h>

#include "clf/QuadraticCostOptimizer.hpp"

#include "TestCostFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(MinimizeQuadraticCostTests, DenseLU) {
  auto cost = std::make_shared<tests::DenseQuadraticCostTest>();

  pt::ptree pt;
  auto opt = std::make_shared<DenseQuadraticCostOptimizer>(cost, pt);

  EXPECT_TRUE(false);
}

TEST(MinimizeQuadraticCostTests, DenseQR) {
  auto cost = std::make_shared<tests::DenseQuadraticCostTest>();

  EXPECT_TRUE(false);
}

TEST(MinimizeQuadraticCostTests, SparseLU) {
  auto cost = std::make_shared<tests::SparseQuadraticCostTest>();

  pt::ptree pt;
  //auto lm = std::make_shared<SparseLevenbergMarquardt>(cost, pt);

  EXPECT_TRUE(false);
}

TEST(MinimizeQuadraticCostTests, SparseQR) {
  auto cost = std::make_shared<tests::SparseQuadraticCostTest>();

  pt::ptree pt;
  //auto lm = std::make_shared<SparseLevenbergMarquardt>(cost, pt);

  EXPECT_TRUE(false);
}
