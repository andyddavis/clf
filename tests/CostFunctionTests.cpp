#include <gtest/gtest.h>

#include "clf/CostFunction.hpp"

#include "TestCostFunctions.hpp"

using namespace clf;

TEST(CostFunctionTests, DenseMatrices) {
  std::size_t indim = 8;
  tests::DenseCostFunctionTest cost(indim);
  EXPECT_EQ(cost.indim, indim);
}

TEST(CostFunctionTests, SparseMatrices) {
  std::size_t indim = 13;
  tests::SparseCostFunctionTest cost(indim);
  EXPECT_EQ(cost.indim, indim);
}
