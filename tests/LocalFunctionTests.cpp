#include <gtest/gtest.h>

#include "clf/LocalFunction.hpp"
#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(LocalFunctionTests, Construction) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim);

  LocalFunction func(mat);
  EXPECT_EQ(func.InputDimension(), indim);
  EXPECT_EQ(func.OutputDimension(), outdim);
  EXPECT_EQ(func.NumCoefficients(), mat->numBasisFunctions);

  // the coefficients should initialize to zero so the function is zero 
  const Eigen::VectorXd eval = func.Evaluate(Eigen::VectorXd::Random(indim));
  EXPECT_NEAR(eval.norm(), 0.0, 1.0e-14);
}
