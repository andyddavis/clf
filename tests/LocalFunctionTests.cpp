#include <gtest/gtest.h>

#include "clf/LocalFunction.hpp"
#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(LocalFunctionTests, Construction) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.1;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis, delta, xbar);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim);

  LocalFunction func(mat);
  EXPECT_EQ(func.InputDimension(), indim);
  EXPECT_EQ(func.OutputDimension(), outdim);
  EXPECT_EQ(func.NumCoefficients(), mat->numBasisFunctions);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func.NumCoefficients());
  const Eigen::VectorXd eval = func.Evaluate(x, coeff);
  EXPECT_NEAR((eval-mat->ApplyTranspose(x, coeff)).norm(), 0.0, 1.0e-14);
}
