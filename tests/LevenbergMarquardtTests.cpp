#include <gtest/gtest.h>

#include "clf/LevenbergMarquardt.hpp"

using namespace clf;

TEST(LevenbergMarquardtTests, DenseMatrices) {
  auto para = std::shared_ptr<Parameters>();

  DenseLevenbergMarquardt lm(para);

  EXPECT_TRUE(false);
}

TEST(LevenbergMarquardtTests, SparseMatrices) {
  auto para = std::shared_ptr<Parameters>();

  SparseLevenbergMarquardt lm(para);

  EXPECT_TRUE(false);
}
